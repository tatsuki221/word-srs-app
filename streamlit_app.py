import os, io, time, json, base64
import streamlit as st
from PIL import Image
from openai import OpenAI

# ★ ここに“復習エンジンの完全版プロンプト”を貼る
SRS_SYSTEM_PROMPT = """<<YOUR_SRS_PROMPT_HERE>>"""

def now_ms(): return int(time.time()*1000)

# OpenAI クライアント（Cloud の Secrets から読み込む）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="単語SRS", page_icon="📚", layout="wide")
st.title("📚 単語SRS（写真→自動出題）")

# --- セッションストレージ ---
if "WORDS" not in st.session_state: st.session_state.WORDS = []
if "CARDS" not in st.session_state: st.session_state.CARDS = []
if "DUE"   not in st.session_state: st.session_state.DUE = []
if "ANS"   not in st.session_state: st.session_state.ANS = []

# --- OpenAI Vision を使って OCR（画像そのものを送る） ---
def ocr_with_openai(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    msg = [
        {"role":"system","content":"Extract plain text from the image. Return only raw text."},
        {"role":"user","content":[
            {"type":"input_text","text":"Please OCR this image and return plain text only."},
            {"type":"input_image","image_url":{"url":f"data:image/png;base64,{b64}"}}]}
    ]
    resp = client.chat.completions.create(model="gpt-4o-mini", temperature=0, messages=msg)
    return resp.choices[0].message.content.strip()

# --- LLMをJSONで呼ぶユーティリティ ---
def llm_json(system_prompt: str, payload: dict) -> dict:
    r = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":json.dumps(payload, ensure_ascii=False)}]
    )
    txt = r.choices[0].message.content.strip()
    try:
        return json.loads(txt)
    except:
        return {"mode":"serve","session":{"served_at":now_ms(),"items":[]}}

# --- OCR→単語抽出→カード雛形（簡易） ---
def bootstrap_from_text(text: str):
    words = []
    for line in text.splitlines():
        token = line.strip().split(" ")[0]
        if token.isalpha() and 2 <= len(token) <= 20:
            words.append(token.lower())
    words = list(dict.fromkeys(words))
    st.session_state.WORDS += [{"headword": w, "senses": [], "contrast_pairs": []} for w in words]
    now = now_ms()
    for w in words:
        st.session_state.CARDS.append({
            "id": f"c_{w}_{now}",
            "word": w, "stage": 1, "type": "en2ja",
            "prompt": f"【和訳】{w}", "answer": "",
            "tags": {"sense_id": None}, "due_at": now, "last_result": None
        })

# --- serve / grade ---
CFG = {"algo":"leitner","leitner_offsets_days":[1,3,7,14,30],
       "wrong_delay_hours":12,"hard_delay_hours":24,
       "session_max":20,"min_mix_ratio":{"ja2en":0.25,"en2ja":0.25,"cloze":0.25,"contrast":0.25},
       "random_seed":42,"accept_spelling_distance":1,"accept_lemma":True,
       "accept_synonym_if_same_sense":True,"lang":"ja"}

def serve():
    out = llm_json(SRS_SYSTEM_PROMPT, {"now": now_ms(), "config": CFG,
                                       "words": st.session_state.WORDS,
                                       "cards": st.session_state.CARDS})
    st.session_state.DUE = out.get("session",{}).get("items",[])

def grade():
    payload = {"now": now_ms(), "config": CFG, "words": st.session_state.WORDS,
               "cards": st.session_state.CARDS, "user_answers": st.session_state.ANS}
    out = llm_json(SRS_SYSTEM_PROMPT, payload)
    results = out.get("results",[])
    cards = {c["id"]: c for c in st.session_state.CARDS}
    for r in results:
        cid = r["card_id"]
        if cid in cards:
            cards[cid]["stage"] = r["next"]["stage"]
            cards[cid]["due_at"] = r["next"]["due_at"]
            cards[cid]["last_result"] = r["result"]
            for f in r.get("followups",[]):
                st.session_state.CARDS.append({
                    "id": f"fu_{int(time.time()*1000)}",
                    "word": cards[cid]["word"],
                    "stage": max(1, cards[cid]["stage"]-1),
                    "type": f["type"], "prompt": f["prompt"], "answer": f["answer"],
                    "tags": f.get("tags",{}), "due_at": now_ms()+3600*1000, "last_result": None
                })
    st.session_state.ANS = []
    st.success("採点完了・次回スケジュール更新")

# --- UI ---
tab1, tab2, tab3 = st.tabs(["1) 写真取り込み", "2) 今日の出題", "3) データ"])
with tab1:
    st.write("画像をアップロード or カメラ撮影")
    img_file = st.file_uploader("画像を選択", type=["png","jpg","jpeg"])
    cam = st.camera_input("📷 カメラで撮る（スマホでも可）")
    if img_file or cam:
        img = Image.open(img_file or cam).convert("RGB")
        st.image(img, use_column_width=True)
        if st.button("OCRしてカード作成"):
            text = ocr_with_openai(img)
            st.text_area("OCR結果", text, height=200)
            bootstrap_from_text(text)
            st.success("カード作成完了 → 『2) 今日の出題』へ")

with tab2:
    c1, c2 = st.columns(2)
    if c1.button("今日の出題（serve）"): serve()
    if c2.button("採点（grade）"): grade()

    st.write("---")
    for it in st.session_state.DUE:
        st.markdown(f"**[{it['type']}] Stage{it['stage']}**")
        st.write(it["prompt"])
        ans = st.text_input(f"解答（{it['card_id']}）", key=f"ans_{it['card_id']}")
        if ans:
            st.session_state.ANS.append({"card_id": it["card_id"], "user_input": ans, "latency_ms": 5000})

with tab3:
    st.write("カード総数:", len(st.session_state.CARDS))
    st.json({"WORDS_sample": st.session_state.WORDS[:5]})
    st.json({"CARDS_sample": st.session_state.CARDS[:5]})
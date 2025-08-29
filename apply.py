import os, time, json, base64, io
from datetime import datetime, timedelta
from typing import List, Dict, Any
import streamlit as st
from PIL import Image
import pytesseract

# ========== LLM 呼び出し（OpenAI） ==========
# 実運用では gpt-5 / gpt-4o などに差し替え
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 復習エンジン・プロンプト（あなたが作った「完全版」）をここに格納
SRS_SYSTEM_PROMPT = r"""
あなたは「英単語学習の復習エンジン」です。日本語・英語の文脈使い分けと多義語判定を重視し、間隔反復（SRS）を高精度で運用します。
以降の会話では、指定した入力スキーマに従って出力のみを返してください。説明文は不要です。
（中略：あなたの“完全版”を丸ごとここに貼り付け）
【開始】
入力に "user_answers" が無いときは serve、あるときは grade を返してください。
"""

# ========== 簡易ストレージ（メモリDB：初回はこれでOK。後でSQLiteに置換） ==========
if "WORDS" not in st.session_state: st.session_state.WORDS = []      # 語彙メタ
if "CARDS" not in st.session_state: st.session_state.CARDS = []      # 出題カード
if "DUE_NOW" not in st.session_state: st.session_state.DUE_NOW = []  # 今回セッション出題
if "ANSWERS" not in st.session_state: st.session_state.ANSWERS = []  # ユーザ解答バッファ

def now_ms(): return int(time.time()*1000)

# ========== OCR ==========
def ocr_image(img: Image.Image) -> str:
    # 言語モデルは英+日。tesseract の追加言語パックが必要な場合あり（jpn）
    try:
        text = pytesseract.image_to_string(img, lang="eng+jpn")
    except:
        text = pytesseract.image_to_string(img, lang="eng")
    return text

# ========== LLMユーティリティ ==========
def llm_json(system_prompt: str, user_json: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # 例。必要に応じ変更
        temperature=0.0,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":json.dumps(user_json, ensure_ascii=False)}
        ]
    )
    txt = resp.choices[0].message.content.strip()
    # 念のためガード
    try:
        return json.loads(txt)
    except:
        # 失敗時は空の規定形
        return {"mode":"serve","session":{"served_at":now_ms(),"items":[]}}

# ========== カード生成（MVP：OCR結果→見出し語抽出→カード雛形） ==========
def bootstrap_cards_from_text(raw_text: str):
    # シンプルに行ごとに英単語らしきものを拾う（実務では正規化を強化）
    words = []
    for line in raw_text.splitlines():
        token = line.strip().split(" ")[0]
        if token.isalpha() and len(token) <= 20:
            words.append(token.lower())
    words = list(dict.fromkeys(words))  # 重複排除
    # words → 最低限のカードを作る（Stage1）
    new_cards = []
    for w in words:
        card = {
            "id": f"c_{w}_{int(time.time()*1000)}",
            "word": w,
            "stage": 1,
            "type": "en2ja",                     # 初見は意味輪郭づくり
            "prompt": f"【和訳】{w}",
            "answer": "",                        # 採点は後続LLMで
            "tags": {"sense_id": None},
            "due_at": now_ms(),
            "last_result": None
        }
        new_cards.append(card)
    st.session_state.WORDS += [{"headword": w, "senses": [], "contrast_pairs": []} for w in words]
    st.session_state.CARDS += new_cards

# ========== SRS 出題（serve） ==========
def serve_session():
    cfg = {
        "algo":"leitner",
        "leitner_offsets_days":[1,3,7,14,30],
        "wrong_delay_hours":12,
        "hard_delay_hours":24,
        "session_max":20,
        "min_mix_ratio":{"ja2en":0.25,"en2ja":0.25,"cloze":0.25,"contrast":0.25},
        "random_seed":42,
        "accept_spelling_distance":1,
        "accept_lemma":True,
        "accept_synonym_if_same_sense":True,
        "lang":"ja"
    }
    payload = {
        "now": now_ms(),
        "config": cfg,
        "words": st.session_state.WORDS,
        "cards": st.session_state.CARDS
    }
    out = llm_json(SRS_SYSTEM_PROMPT, payload)
    items = out.get("session",{}).get("items",[])
    st.session_state.DUE_NOW = items
    st.session_state.ANSWERS = []  # リセット

# ========== 採点（grade） ==========
def grade_session():
    ua = st.session_state.ANSWERS
    # LLM に丸ごと渡して採点＆スケジュール更新を受け取る
    cfg = {
        "algo":"leitner",
        "leitner_offsets_days":[1,3,7,14,30],
        "wrong_delay_hours":12,
        "hard_delay_hours":24,
        "session_max":20,
        "min_mix_ratio":{"ja2en":0.25,"en2ja":0.25,"cloze":0.25,"contrast":0.25},
        "random_seed":42,
        "accept_spelling_distance":1,
        "accept_lemma":True,
        "accept_synonym_if_same_sense":True,
        "lang":"ja"
    }
    payload = {
        "now": now_ms(),
        "config": cfg,
        "words": st.session_state.WORDS,
        "cards": st.session_state.CARDS,
        "user_answers": ua
    }
    out = llm_json(SRS_SYSTEM_PROMPT, payload)  # mode: grade
    results = out.get("results",[])
    # ローカルカードを更新（stage/due_at）
    card_by_id = {c["id"]: c for c in st.session_state.CARDS}
    for r in results:
        cid = r["card_id"]
        if cid in card_by_id:
            card_by_id[cid]["stage"] = r["next"]["stage"]
            card_by_id[cid]["due_at"] = r["next"]["due_at"]
            card_by_id[cid]["last_result"] = r["result"]
            # フォローアップをカード化して追加
            for f in r.get("followups", []):
                st.session_state.CARDS.append({
                    "id": f"fu_{int(time.time()*1000)}",
                    "word": card_by_id[cid]["word"],
                    "stage": max(1, card_by_id[cid]["stage"]-1),
                    "type": f["type"],
                    "prompt": f["prompt"],
                    "answer": f["answer"],
                    "tags": f.get("tags", {}),
                    "due_at": now_ms() + 3600*1000,  # 1時間後
                    "last_result": None
                })
    st.success("採点・スケジュール更新が完了しました。")

# ========== UI ==========
st.set_page_config(page_title="単語SRS（写真→自動出題）", page_icon="📚", layout="wide")
st.title("📚 単語SRS（写真→自動出題）")

tab1, tab2, tab3 = st.tabs(["1) 取り込み（写真）", "2) 今日の出題", "3) データ"])

with tab1:
    st.subheader("単語帳のページを撮影してアップロード")
    img_file = st.file_uploader("画像ファイル（JPG/PNG）", type=["png","jpg","jpeg"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="アップロード画像", use_column_width=True)
        if st.button("OCR → 単語抽出 → カード作成"):
            text = ocr_image(img)
            st.text_area("OCR結果", text, height=200)
            bootstrap_cards_from_text(text)
            st.success("カードを作成しました。『2) 今日の出題』へ。")

with tab2:
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("今日の出題を開始（serve）"):
            serve_session()
    with colB:
        if st.button("解答を送って採点（grade）"):
            grade_session()

    st.write("---")
    for item in st.session_state.DUE_NOW:
        st.markdown(f"**[{item['type']}] Stage{item['stage']}**")
        st.write(item["prompt"])
        ans = st.text_input(f"解答（card_id={item['card_id']}）", key=f"ans_{item['card_id']}")
        if ans:
            st.session_state.ANSWERS.append({
                "card_id": item["card_id"],
                "user_input": ans,
                "latency_ms": 5000
            })

with tab3:
    st.write("**カード総数**:", len(st.session_state.CARDS))
    st.json({"WORDS": st.session_state.WORDS[:5]})
    st.json({"CARDS_sample": st.session_state.CARDS[:5]})
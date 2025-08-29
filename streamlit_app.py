# streamlit_app.py
import os, io, time, json, base64, random
from typing import Dict, Any, List
import streamlit as st
from PIL import Image
from pillow_heif import register_heif_opener
from openai import OpenAI

# HEIC(HEIF) を Pillow で開けるように登録
register_heif_opener()

# ==============================
# 1) SRS 復習エンジン（完全版プロンプト）
# ==============================
SRS_SYSTEM_PROMPT = r"""
あなたは「英単語学習の復習エンジン」です。日本語・英語の文脈使い分けと多義語判定を重視し、間隔反復（SRS）を高精度で運用します。
以降の会話では、指定した入力スキーマに従って出力のみを返してください。説明文は不要です。

========================
【目的】
- 期限になったカードを最適順で出題
- 回数（Stage）に応じて問題形式を自動切替
- 多義語の意味分岐と近義語の文脈比較を厳密判定
- 採点は「語形・語義・文脈」の3層で評価
- 結果に応じて次回復習時刻とフォローアップを即時決定

========================
【前提・用語】
- カード：1語につき複数（ja->en / en->ja / cloze / contrast）
- Stage（1..5）: 出題難度レベル。正解で+1、誤答で-1、hardで据え置き
- due_at：次回出題の時刻（UTC epoch ms）
- deck：Leitner箱（1..5）。Stageと同義で扱ってよい
- sense_id：多義語の特定義（例 "issue#2"）

========================
【入力スキーマ】（JSON）
{
  "now": <number epoch_ms>,
  "config": {
    "algo": "leitner|sm2",
    "leitner_offsets_days": [1,3,7,14,30],
    "wrong_delay_hours": 12,
    "hard_delay_hours": 24,
    "session_max": 40,
    "min_mix_ratio": {"ja2en":0.25,"en2ja":0.25,"cloze":0.25,"contrast":0.25},
    "random_seed": 42,
    "accept_spelling_distance": 1,
    "accept_lemma": true,
    "accept_synonym_if_same_sense": true,
    "lang": "ja"
  },
  "words": [
    {
      "headword": "deal",
      "senses": [
        {"sense_id":"deal#handle","core_jp":"扱う/処理する","frames":["deal with + NP"],"collocations":["deal with a problem"],"register":"neutral"},
        {"sense_id":"deal#distribute","core_jp":"分配する","frames":["deal A to B"],"register":"neutral"}
      ],
      "contrast_pairs": [
        {"a":"deal with","b":"cope with","meaning_delta":"処理vs耐える","collocation_delta":"tasks/issues vs difficulties/stress","register_delta":"中立vs苦境ニュアンス"}
      ]
    }
  ],
  "cards": [
    {
      "id":"c1",
      "word":"deal with",
      "stage":2,
      "type":"ja2en",
      "prompt":"私はこの問題にすぐ対処した。",
      "answer":"deal with",
      "tags":{"sense_id":"deal#handle"},
      "due_at": 1735400000000,
      "last_result":"correct|wrong|hard|null"
    }
  ],
  "user_answers": [
    {
      "card_id":"c1",
      "user_input":"cope with",
      "latency_ms": 6500
    }
  ]
}

========================
【出力モード】
- セッション開始（問題配布）: "mode":"serve"
- 採点＋次回更新: "mode":"grade"
呼び分けは、入力に "user_answers" が無ければ serve、有れば grade とする。

========================
【出力スキーマ】

■ serve（出題配布）
{
  "mode":"serve",
  "session": {
    "served_at": <epoch_ms>,
    "items":[
      {
        "card_id":"c1",
        "stage":2,
        "type":"ja2en|en2ja|cloze|contrast|compose",
        "prompt":"...",
        "options":["A","B","C"]|null,
        "meta":{
          "word":"deal with",
          "sense_id":"deal#handle"|null,
          "signals":["with + NP","problem"]
        }
      }
    ]
  }
}

■ grade（採点・更新）
{
  "mode":"grade",
  "results":[
    {
      "card_id":"c1",
      "result":"correct|wrong|hard",
      "score": 0.0..1.0,
      "rubric": {
        "form": "exact|lemma|typo|wrong_spelling",
        "sense": "match|mismatch|unknown",
        "context": "natural|awkward|conflict",
        "register": "ok|mismatch"
      },
      "explanation": "なぜその判定か（日本語）",
      "next": {"stage": <1..5>,"due_at": <epoch_ms>},
      "followups": [
        {
          "type":"contrast|cloze|ja2en|en2ja|micro_drill",
          "prompt":"...",
          "answer":"...",
          "tags":{"reason":"sense_mismatch|prep_error|near_syn_confusion","sense_id":"..."}
        }
      ],
      "log": {"latency_ms": 6500}
    }
  ]
}

========================
【カード選定（serve のロジック）】
1) due_at <= now のカードのみ対象。最大 session_max 件。
2) シャッフルは random_seed を用い、同一 sense_id が連続しないよう分散。
3) 出題比率ルール（不足タイプは優先補充）:
   - Stage1: ja2en/en2ja/cloze ≒ 2:1:2
   - Stage2: ja2en/en2ja/cloze ≒ 1:2:2
   - Stage3: contrast/cloze中心
   - Stage4: compose/2空所cloze/誤答誘発
   - Stage5: ミックス模試 + 理由説明
4) contrast は毎セッションで最低2問（可能なら）。
5) 同一語の別義（sense_id）が存在する場合、連続出題を避ける。

========================
【採点規則（grade のロジック）】
- 三層評価：①語形（form）②語義（sense）③文脈（context）
  A) form: exact / lemma / typo / wrong_spelling
  B) sense: 同語でも tags.sense_id と不一致なら mismatch
  C) context: frames/collocation/registerの矛盾
  D) register: 求められた語感ズレは減点
- スコア例:
  exact+match+natural → 1.00
  lemma+match+natural → 0.95
  typo+match+natural → 0.90
  exact+match+awkward → 0.85
  exact+sense_mismatch → 0.40
  wrong_spelling or context_conflict → 0.00〜0.20
- ラベル:
  score >= 0.90 → "correct"
  0.60 <= score < 0.90 → "hard"
  score < 0.60 → "wrong"

========================
【次回スケジューリング】
- algo="leitner":
  correct → stage+1 / due = now + offsets_days[stage-1]
  hard    → stage据置 / due = now + hard_delay_hours
  wrong   → stage-1 / due = now + wrong_delay_hours
- algo="sm2"（簡易）もサポート

========================
【フォローアップ生成規則】
- sense_mismatch: 同語別義のcloze 2問 + en2ja 1問
- near_syn_confusion: contrast 2問 + ダブル空所cloze 1問
- prep/frame_error: micro_drill 3問
- register_mismatch: レジスター置換 2問

========================
【出題生成（serve時の文面ルール）】
- 穴埋めは ____ を使用
- contrast は理由説明を促す一文を含める
- meta.signals にコロケ手掛かりを列挙

========================
【出力の決定性】
- random_seed に基づいて決定的
- 温度は0相当

========================
【バリデーション】
- serve: items が空なら空配列
- grade: 不明card_idは無視
- JSON厳格

========================
【開始】
入力に "user_answers" が無ければ serve、有れば grade を返す。
"""

# ==============================
# 2) OpenAI クライアント（Secrets/環境変数対応）
# ==============================
def _get_api_key():
    return os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

_api_key = _get_api_key()
if not _api_key:
    st.set_page_config(page_title="単語SRS（設定エラー）", page_icon="⚠️")
    st.error("OPENAI_API_KEY が見つかりません。Streamlit Cloud の『Settings → Secrets』に OPENAI_API_KEY を設定してください。")
    st.stop()

client = OpenAI(api_key=_api_key)

def now_ms() -> int:
    return int(time.time() * 1000)

# ==============================
# 3) OpenAI Vision OCR（Tesseract不要）
# ==============================
def ocr_with_openai(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    messages = [
        {"role":"system","content":"Extract plain text from the image. Return only raw text."},
        {"role":"user","content":[
            {"type":"input_text","text":"Please OCR this image and return plain text only."},
            {"type":"input_image","image_url":{"url":f"data:image/png;base64,{b64}"}}
        ]}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=messages
    )
    return resp.choices[0].message.content.strip()

# ==============================
# 4) LLM JSON ユーティリティ
# ==============================
def llm_json(system_prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":json.dumps(payload, ensure_ascii=False)}
        ]
    )
    txt = resp.choices[0].message.content.strip()
    try:
        return json.loads(txt)
    except Exception:
        return {"mode":"serve","session":{"served_at":now_ms(),"items":[]}}

# ==============================
# 5) 画像オープン（HEIC対応）
# ==============================
def _open_uploaded_image(uploaded) -> Image.Image:
    # Streamlit の UploadedFile は bytes を返せる
    data = uploaded.getvalue()
    img = Image.open(io.BytesIO(data))
    return img.convert("RGB")

# ==============================
# 6) 簡易カード生成（OCRテキスト→語群→カード雛形）
# ==============================
def bootstrap_from_text(text: str):
    words = []
    for line in text.splitlines():
        token = line.strip().split(" ")[0]
        if token.isalpha() and 2 <= len(token) <= 20:
            words.append(token.lower())
    words = list(dict.fromkeys(words))
    t = now_ms()
    st.session_state.WORDS += [{"headword": w, "senses": [], "contrast_pairs": []} for w in words]
    for w in words:
        st.session_state.CARDS.append({
            "id": f"c_{w}_{t}_{random.randint(100,999)}",
            "word": w,
            "stage": 1,
            "type": "en2ja",
            "prompt": f"【和訳】{w}",
            "answer": "",
            "tags": {"sense_id": None},
            "due_at": t,
            "last_result": None
        })

# ==============================
# 7) セッション状態（疑似DB）
# ==============================
st.set_page_config(page_title="単語SRS（写真→自動出題）", page_icon="📚", layout="wide")
st.title("📚 単語SRS（写真→自動出題 / Streamlit Cloud 版）")

if "WORDS" not in st.session_state: st.session_state.WORDS = []
if "CARDS" not in st.session_state: st.session_state.CARDS = []
if "DUE"   not in st.session_state: st.session_state.DUE   = []
if "ANS"   not in st.session_state: st.session_state.ANS   = []

# SRS 設定
CFG = {
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

# ==============================
# 8) serve / grade ロジック
# ==============================
def serve_session():
    out = llm_json(SRS_SYSTEM_PROMPT, {
        "now": now_ms(),
        "config": CFG,
        "words": st.session_state.WORDS,
        "cards": st.session_state.CARDS
    })
    st.session_state.DUE = out.get("session",{}).get("items",[])

def grade_session():
    payload = {
        "now": now_ms(),
        "config": CFG,
        "words": st.session_state.WORDS,
        "cards": st.session_state.CARDS,
        "user_answers": st.session_state.ANS
    }
    out = llm_json(SRS_SYSTEM_PROMPT, payload)
    results = out.get("results",[])
    card_map = {c["id"]: c for c in st.session_state.CARDS}
    for r in results:
        cid = r.get("card_id")
        if cid in card_map:
            card = card_map[cid]
            nxt = r.get("next",{})
            card["stage"] = nxt.get("stage", card["stage"])
            card["due_at"] = nxt.get("due_at", card["due_at"])
            card["last_result"] = r.get("result", card.get("last_result"))
            # フォローアップをカード化
            for f in r.get("followups",[]):
                st.session_state.CARDS.append({
                    "id": f"fu_{int(time.time()*1000)}",
                    "word": card["word"],
                    "stage": max(1, card["stage"]-1),
                    "type": f.get("type","cloze"),
                    "prompt": f.get("prompt",""),
                    "answer": f.get("answer",""),
                    "tags": f.get("tags",{}),
                    "due_at": now_ms()+3600*1000,
                    "last_result": None
                })
    st.session_state.ANS = []
    st.success("採点完了・次回スケジュール更新")

# ==============================
# 9) データの保存/読み込み（JSON）
# ==============================
def export_json() -> str:
    data = {"words": st.session_state.WORDS, "cards": st.session_state.CARDS}
    return json.dumps(data, ensure_ascii=False, indent=2)

def import_json(txt: str):
    try:
        data = json.loads(txt)
        st.session_state.WORDS = data.get("words", [])
        st.session_state.CARDS = data.get("cards", [])
        st.success("JSONを読み込みました。")
    except Exception as e:
        st.error(f"JSONの読み込みに失敗: {e}")

# ==============================
# 10) UI
# ==============================
tab1, tab2, tab3, tab4 = st.tabs(["1) 写真取り込み", "2) 今日の出題", "3) データ", "4) 設定"])

with tab1:
    st.subheader("画像をアップロード or カメラ撮影（スマホOK）")

    # ✅ カメラ常時起動を防ぐ：トグルで表示切替
    st.session_state.setdefault("use_cam", False)
    st.session_state.use_cam = st.toggle("📷 カメラを使う", value=st.session_state.use_cam)

    col1, col2 = st.columns(2)
    with col1:
        # ✅ HEICも受け付ける
        img_file = st.file_uploader("画像を選択（JPG/PNG/HEIC）", type=["png","jpg","jpeg","heic"])
    with col2:
        cam = st.camera_input("カメラで撮る", key="cam_input") if st.session_state.use_cam else None

    # ファイルがあれば優先。どちらもNoneなら何もしない
    uploaded = img_file or cam
    if uploaded:
        try:
            img = _open_uploaded_image(uploaded)
            st.image(img, caption="プレビュー", use_column_width=True)
            if st.button("OCRしてカード作成", type="primary"):
                with st.spinner("OCR中…"):
                    text = ocr_with_openai(img)
                st.text_area("OCR結果（編集OK）", text, height=200, key="OCR_TEXT")
                if st.button("↑ このテキストからカード作成"):
                    bootstrap_from_text(st.session_state.get("OCR_TEXT",""))
                    st.success("カードを作成しました → 『2) 今日の出題』へ")
        except Exception as e:
            st.error("画像を開けませんでした（形式未対応/破損の可能性）。別形式で試すか、もう一度撮影してください。")
            st.caption(f"詳細: {e}")

with tab2:
    st.subheader("今日の出題")
    c1, c2, c3 = st.columns([1,1,1])
    if c1.button("今日の出題（serve）", type="primary"):
        serve_session()
    if c2.button("採点（grade）", type="secondary"):
        grade_session()
    if c3.button("解答リセット"):
        st.session_state.ANS = []

    st.write("---")
    if not st.session_state.DUE:
        st.info("出題キューが空です。カードの due_at を満たすと表示されます。")
    for it in st.session_state.DUE:
        st.markdown(f"**[{it.get('type','')}] Stage {it.get('stage','?')}**")
        st.write(it.get("prompt",""))
        ans = st.text_input(
            f"解答（card_id={it.get('card_id')}）",
            key=f"ans_{it.get('card_id')}"
        )
        if ans:
            # 既存解答があれば置換
            st.session_state.ANS = [a for a in st.session_state.ANS if a["card_id"] != it["card_id"]]
            st.session_state.ANS.append({"card_id": it["card_id"], "user_input": ans, "latency_ms": 5000})

with tab3:
    st.subheader("データの確認・バックアップ")
    st.write("カード総数:", len(st.session_state.CARDS))
    st.json({"WORDS_sample": st.session_state.WORDS[:5]})
    st.json({"CARDS_sample": st.session_state.CARDS[:5]})

    st.write("——")
    st.download_button(
        "📥 JSONエクスポート",
        data=export_json().encode("utf-8"),
        file_name="word_srs_data.json",
        mime="application/json"
    )
    st.write("——")
    up = st.file_uploader("📤 JSONインポート（エクスポートしたファイルを選択）", type=["json"], key="json_in")
    if up:
        txt = up.read().decode("utf-8")
        if st.button("JSONを読み込む"):
            import_json(txt)

with tab4:
    st.subheader("設定（SRSパラメータ）")
    st.caption("※ 変更後は出題/採点のたびに反映されます。")
    CFG["algo"] = st.selectbox("アルゴリズム", ["leitner","sm2"], index=0)
    CFG["session_max"] = st.slider("1セッションの最大出題数", 5, 50, CFG["session_max"])
    CFG["wrong_delay_hours"] = st.slider("誤答の遅延（時間）", 1, 48, CFG["wrong_delay_hours"])
    CFG["hard_delay_hours"] = st.slider("Hardの遅延（時間）", 1, 48, CFG["hard_delay_hours"])
    st.write("間隔（days）:", CFG["leitner_offsets_days"])

    st.write("—— 開発者向け ——")
    st.code("Secrets に OPENAI_API_KEY を設定してください。", language="bash")
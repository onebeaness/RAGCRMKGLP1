# -*- coding: utf-8 -*-
"""GLP-1 검색증강 질의응답 데모 — 응답 유보 및 미응답 질의 수집 포함.

안전 설계: 검색 임계값 게이트를 통과한 질문에만 LLM(Gemini)이 답을 합성하고,
그때도 게이트를 통과한 문서 내용만 근거로 쓴다. 게이트에서 떨어지면 LLM을
호출하지 않고 의료진 검토 대기열로 보낸다. API 키가 없거나 호출이 실패하면
검색 결과 원문 표시로 자동 폴백한다.

키 설정:  app/.streamlit/secrets.toml 에  GEMINI_API_KEY = "..."  (git 미포함)

실행:  streamlit run demo.py
"""
import json
import os
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def load_embedder():
    """임베딩 모델이 설치돼 있으면 의미 검색, 없으면 TF-IDF 폴백 (이중 모드)."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None


def compute_sims(question, texts):
    """하이브리드 검색: 의미(임베딩) + 표면 단어 일치(TF-IDF) 합산.
    임베딩 모델이 없으면 TF-IDF 단독 폴백."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(texts + [question])
    tfidf = cosine_similarity(vec.transform([question]), vec.transform(texts))[0]
    model = load_embedder()
    if model is not None:
        emb = model.encode(texts + [question], normalize_embeddings=True)
        sem = emb[-1] @ emb[:-1].T
        return 0.5 * sem + 0.5 * tfidf, "하이브리드 (임베딩 + TF-IDF)", 0.23
    return tfidf, "TF-IDF (문자 n-gram 폴백)", 0.15


def gemini_key():
    """secrets.toml → 환경변수 순서로 키를 찾는다. 없으면 None (검색 전용 모드)."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def synthesize_answer(question, docs):
    """게이트를 통과한 문서만 근거로 Gemini가 답변을 합성한다.
    실패하거나 문서 근거가 부족하면 None을 반환해 원문 표시로 폴백한다."""
    key = gemini_key()
    if not key:
        return None
    context = "\n\n".join(
        f"[{i+1}] {d['title']}\n{d['text']}\n(출처: {d['source']})"
        for i, d in enumerate(docs)
    )
    prompt = (
        "당신은 GLP-1 비만치료제 클리닉의 복약 안내 도우미입니다.\n"
        "아래 '공식 문서'의 내용만 근거로 환자의 질문에 답하십시오.\n"
        "규칙:\n"
        "1. 문서에 없는 내용은 한 문장도 추가하지 마십시오.\n"
        "2. 문서만으로 답할 수 없으면 정확히 '문서근거부족' 네 글자만 출력하십시오.\n"
        "3. 2~4문장, 존댓말. 진단·처방 변경 권고는 하지 말고 필요하면 '처방 의사와 상담'을 안내하십시오.\n"
        "4. 근거로 쓴 문서 번호를 문장 끝에 [1]처럼 표기하십시오.\n\n"
        f"공식 문서:\n{context}\n\n환자 질문: {question}"
    )
    try:
        r = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent",
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 2048,
                },
            },
            timeout=25,
        )
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if not text or "문서근거부족" in text:
            return None
        return text
    except Exception:
        return None

HERE = Path(__file__).resolve().parent
CORPUS_F = HERE / "corpus.json"
QUEUE_F = HERE / "failed_queue.json"
FIG = HERE.parent / "notebooks" / "figures" / "activity_survival.png"


def load_json(p, default):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return default


def save_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


corpus = load_json(CORPUS_F, [])
queue = load_json(QUEUE_F, [])

st.set_page_config(page_title="GLP-1 질의응답 데모", layout="wide")
st.title("GLP-1 복약 순응도 관리 — 검색증강 질의응답 시스템 (데모)")
st.caption(
    "데모입니다. 의료 조언이 아닙니다. 답변은 공식 문서 요약(코퍼스)을 근거로만 합성·표시되며, "
    "코퍼스에 없는 질문은 답하지 않고 의료진 검토 대기열에 쌓입니다."
)

with st.sidebar:
    st.subheader("시스템 상태")
    st.metric("코퍼스 문서", f"{len(corpus)}건")
    st.metric("검토 대기 질문", f"{sum(1 for q in queue if q['status']=='대기')}건")
    engine_name = "하이브리드 (임베딩 + TF-IDF)" if load_embedder() else "TF-IDF (문자 n-gram 폴백)"
    default_thr = 0.23 if load_embedder() else 0.15
    st.caption(f"검색 엔진: **{engine_name}**")
    llm_state = "Gemini 연결됨 — 게이트 통과 시 답변 합성" if gemini_key() else "미연결 — 검색 결과 원문 표시"
    st.caption(f"LLM: **{llm_state}**")
    thr = st.slider("검색 실패 임계값 (유사도)", 0.05, 0.70, default_thr, 0.01,
                    help="최고 유사도가 이 값 미만이면 '코퍼스에 답 없음'으로 처리")
    if FIG.exists():
        st.divider()
        st.caption("지속 신호 (게시 활동 잔존 곡선 — 복약 아님)")
        st.image(str(FIG))

tab_patient, tab_admin = st.tabs(["환자 질의응답", "관리자 콘솔 — 미응답 질의 검토"])


# ── 1차 산출: 환자 질의응답 ─────────────────────────────────
with tab_patient:
    q = st.text_input("궁금한 것을 물어보세요", placeholder="예: 메스꺼움은 보통 언제까지 가나요?")
    if st.button("질문", type="primary") and q.strip():
        texts = [c["title"] + " " + c["text"] for c in corpus]
        sims, _engine, _ = compute_sims(q, texts)
        ranked = sorted(zip(sims, corpus), key=lambda x: -x[0])
        best = ranked[0][0] if ranked else 0.0

        if best >= thr:
            hits = [(s, c) for s, c in ranked[:3] if s >= thr * 0.6]
            answer = None
            if gemini_key():
                with st.spinner("게이트 통과 문서를 근거로 답변을 합성하는 중…"):
                    answer = synthesize_answer(q, [c for _, c in hits])
            if answer:
                st.success("공식 문서 근거로 답변을 합성했습니다. 근거 원문은 아래에 있습니다.")
                with st.container(border=True):
                    st.markdown(answer)
                    st.caption("합성: Gemini · 근거는 게이트를 통과한 아래 문서로 제한됨")
            else:
                st.success("공식 문서에서 관련 근거를 찾았습니다. 아래는 원문 요약과 출처입니다.")
            for i, (s, c) in enumerate(hits):
                with st.container(border=True):
                    st.markdown(f"**[{i+1}] {c['title']}**  \n{c['text']}")
                    st.caption(f"출처: {c['source']} · 유사도 {s:.2f}")
            st.caption("이 안내는 공식 문서 요약에 근거하며, 개인별 판단은 처방 의사와 상담하십시오.")
        else:
            st.warning(
                "공식 문서에서 답을 찾지 못했습니다. 이 질문은 **의료진 검토 대기열**에 등록되었고, "
                "검토 후 답변이 코퍼스에 추가되면 같은 질문에 답할 수 있게 됩니다."
            )
            queue.append({
                "question": q.strip(),
                "asked_at": datetime.now().isoformat(timespec="seconds"),
                "best_sim": round(float(best), 3),
                "status": "대기",
                "note": "",
            })
            save_json(QUEUE_F, queue)
            st.caption("→ 시판후 안전정보 후보: 코퍼스에 없는 질의는 미기재 증상·정보 수요의 후보로 축적된다")


# ── 2차 산출: 관리자 검토 및 지식 베이스 갱신 ────────────────
with tab_admin:
    pending = [x for x in queue if x["status"] == "대기"]
    done = [x for x in queue if x["status"] != "대기"]
    st.markdown(f"**검토 대기 {len(pending)}건 · 처리 완료 {len(done)}건**")
    st.caption("검색에 실패한 질문 = 미기재 증상 후보. 실제 운영에서는 이 단계에서 MedDRA 표준화 및 KAERS 대조로 연결된다(설계).")

    if not pending:
        st.info("대기 중인 질문이 없습니다. 환자 질의응답 탭에서 코퍼스에 없는 질문을 입력해 보세요.")
    for i, item in enumerate(pending):
        with st.container(border=True):
            st.markdown(f"**Q. {item['question']}**")
            st.caption(f"접수 {item['asked_at']} · 최고 유사도 {item['best_sim']}")
            title = st.text_input("문서 제목", key=f"t{i}", placeholder="예: 임신 계획 시 투여 중단")
            ans = st.text_area("검토 답변 (공식 문서 근거로 작성)", key=f"a{i}")
            src = st.text_input("출처", key=f"s{i}", placeholder="예: 식약처 허가사항 — 임부 관련 (검토자: OOO)")
            if st.button("코퍼스에 편입 — 지식 베이스 갱신", key=f"b{i}", type="primary"):
                if not (title.strip() and ans.strip() and src.strip()):
                    st.error("제목·답변·출처를 모두 입력해야 편입할 수 있습니다.")
                else:
                    corpus.append({
                        "id": f"loop-{len(corpus)+1:02d}",
                        "title": title.strip(),
                        "text": ans.strip(),
                        "source": src.strip(),
                    })
                    save_json(CORPUS_F, corpus)
                    item["status"] = "편입됨"
                    item["note"] = title.strip()
                    save_json(QUEUE_F, queue)
                    st.rerun()

    if done:
        st.divider()
        st.caption("처리 이력")
        for item in done:
            st.markdown(f"- ~~{item['question']}~~ → **{item['note']}** ({item['status']})")

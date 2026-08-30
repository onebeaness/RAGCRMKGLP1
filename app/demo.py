# -*- coding: utf-8 -*-
"""GLP-1 루프 데모 — 검색 기반 질의응답(앞면) + 실패 질의 대기열(뒷면).

생성(LLM) 없이 검색+원문 표시로만 동작한다: 답변이 코퍼스(허가사항 요약) 밖으로
나갈 수 없다는 안전 설계를 데모 자체로 보여주는 구성. LLM 합성은 문서 III장에 설계로 명시.

실행:  streamlit run demo.py
"""
import json
from datetime import datetime
from pathlib import Path

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

st.set_page_config(page_title="GLP-1 루프 데모", layout="wide")
st.title("GLP-1 루프 — 하나의 루프, 두 개의 얼굴 (데모)")
st.caption(
    "데모입니다. 의료 조언이 아니며, 답변은 공식 문서 요약(코퍼스)의 검색 결과만 표시합니다. "
    "코퍼스에 없는 질문은 답하지 않고 의료진 검토 대기열에 쌓입니다."
)

with st.sidebar:
    st.subheader("루프 상태")
    st.metric("코퍼스 문서", f"{len(corpus)}건")
    st.metric("검토 대기 질문", f"{sum(1 for q in queue if q['status']=='대기')}건")
    engine_name = "하이브리드 (임베딩 + TF-IDF)" if load_embedder() else "TF-IDF (문자 n-gram 폴백)"
    default_thr = 0.23 if load_embedder() else 0.15
    st.caption(f"검색 엔진: **{engine_name}**")
    thr = st.slider("검색 실패 임계값 (유사도)", 0.05, 0.70, default_thr, 0.01,
                    help="최고 유사도가 이 값 미만이면 '코퍼스에 답 없음'으로 처리")
    if FIG.exists():
        st.divider()
        st.caption("지속 신호 (게시 활동 잔존 곡선 — 복약 아님)")
        st.image(str(FIG))

tab_patient, tab_admin = st.tabs(["환자 창구 — 질문하기", "관리자 콘솔 — 실패 질의 검토"])


# ── 앞면: 환자 창구 ──────────────────────────────────────────
with tab_patient:
    q = st.text_input("궁금한 것을 물어보세요", placeholder="예: 메스꺼움은 보통 언제까지 가나요?")
    if st.button("질문", type="primary") and q.strip():
        texts = [c["title"] + " " + c["text"] for c in corpus]
        sims, _engine, _ = compute_sims(q, texts)
        ranked = sorted(zip(sims, corpus), key=lambda x: -x[0])
        best = ranked[0][0] if ranked else 0.0

        if best >= thr:
            st.success("공식 문서에서 관련 근거를 찾았습니다. 아래는 원문 요약과 출처입니다.")
            for s, c in ranked[:3]:
                if s < thr * 0.6:
                    continue
                with st.container(border=True):
                    st.markdown(f"**{c['title']}**  \n{c['text']}")
                    st.caption(f"출처: {c['source']} · 유사도 {s:.2f}")
            st.caption("이 안내는 공식 문서 요약의 검색 결과이며, 개인별 판단은 처방 의사와 상담하십시오.")
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
            st.caption("→ 뒷면(4상)의 원료: 코퍼스에 없는 질문 = 미기재 증상·정보 수요 후보")


# ── 뒷면: 관리자 콘솔 ────────────────────────────────────────
with tab_admin:
    pending = [x for x in queue if x["status"] == "대기"]
    done = [x for x in queue if x["status"] != "대기"]
    st.markdown(f"**검토 대기 {len(pending)}건 · 처리 완료 {len(done)}건**")
    st.caption("검색에 실패한 질문 = 미기재 증상 후보. 실제 서비스에서는 여기서 MedDRA 매핑·KAERS 대조로 이어진다(설계).")

    if not pending:
        st.info("대기 중인 질문이 없습니다. 환자 창구에서 코퍼스에 없는 질문을 해 보세요.")
    for i, item in enumerate(pending):
        with st.container(border=True):
            st.markdown(f"**Q. {item['question']}**")
            st.caption(f"접수 {item['asked_at']} · 최고 유사도 {item['best_sim']}")
            title = st.text_input("문서 제목", key=f"t{i}", placeholder="예: 임신 계획 시 투여 중단")
            ans = st.text_area("검토 답변 (공식 문서 근거로 작성)", key=f"a{i}")
            src = st.text_input("출처", key=f"s{i}", placeholder="예: 식약처 허가사항 — 임부 관련 (검토자: OOO)")
            if st.button("코퍼스에 편입 → 루프 닫기", key=f"b{i}", type="primary"):
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

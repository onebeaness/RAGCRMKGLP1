# RAGCRMKGLP1 — RAG 기반 GLP-1 환자 케어 CRM × 시판후 안전신호 루프

> 환자에게 RAG 기반 응답 CRM), 답하지 못한 질문은 핫라인 보고 (4상).
>
> 논문(조사·연구·구현) 기반 솔루션 개발 과제 · 2026-08-25 ~ 09-01 · 개인 프로젝트 · 윤원빈

---

## OVERVIEW

GLP-1 비만치료제는 시작 환자의 **31.7%가 180일 안에 중단**한다 — 사유 1위는 부작용(26.8%), 2위는 비용(14.4%) (Van Laren, Kaiser EHR, 2026). 문제는 유입이 아니라 **지속**이다.

환자가 중단 전에 무슨 생각을 하는지는 병원 기록에 없다. 그 목소리는 소셜 담론에 있고, Somani et al.(2024)이 Reddit 39만 건을 임베딩·군집화해 **168개 주제 지도**로 그렸다. 이후 15~20편의 연구가 "무엇이 문제인지"까지 규명했지만, 두 가지가 비어 있다(2026-08 실측):

1. 그 위에 세우는 **솔루션 층** — 누구에게, 어떤 개입을, 누구 돈으로
2. **한국어 GLP-1 담론 분석** — 위고비 국내 출시(2024-10) 후 2년째 전무

이 프로젝트는 Somani 파이프라인을 재현하고, 그 위에 **RAG 기반 질의응답(앞면)과 검색 실패 질의의 안전신호 수집(뒷면)** 루프를 설계·시연한다.

## TECHNOLOGY

| 층 | 사용 |
|---|---|
| Language | Python 3.11+ |
| 분석 | sentence-transformers(BGE) · umap-learn · hdbscan · scikit-learn · pandas |
| 데모 | Streamlit · LLM API(토픽 라벨·답변 생성 — 원논문 Llama2-7B 대체, 명시) |
| Data | [Somani et al. Zenodo](https://doi.org/10.5281/zenodo.12209343) — Reddit 391,461건 pickle 212MB + 사전계산 임베딩 (CC-BY 4.0) |

앵커 논문: [Somani et al., Communications Medicine 2024](https://www.nature.com/articles/s43856-024-00566-z) · [원저자 코드](https://github.com/sssomani/glp1_reddit)

## RESULT

### 구현 (동작하는 것)
- [ ] Somani 파이프라인 재현 — 임베딩→UMAP→군집→라벨 (원논문 168토픽 대비 대조표) `docs/` 참조
- [x] **저자별 활동 지속 분포** — 중앙값 65일, 1년 잔존 8.2% *(논문에 없는 신규 산출 — 게시 활동의 대리지표이며 복약 중단이 아님, 한계 명시)*
- [ ] RAG-lite 데모 — 질문 → 코퍼스 검색 → **출처 달린 답변** + 검색 실패 질의 큐 (Streamlit)

### 설계 (문서로 있는 것 — `docs/`)
- 사용자 단위 로그 스키마 (Sehgal 2026의 4기준: 사용자 단위 · MedDRA 표준화 · 정밀도 보고 · DB 교차검증)
- 실패 질의 → 증상 후보 추출 흐름 (Duan 2025 프롬프트 차용) → KAERS 대조 설계 (건양대 2023·24 선례)
- 이중 수익 구조(클리닉 구독 + 제약사 RMP 이행 데이터) · 동의서 2차 활용 조항 선탑재 · 의료행위 경계 명시

> **구현/설계 구분은 의도적입니다.** 일주일 안에 돌아가는 것과 문서로 설계한 것을 섞지 않습니다.

## 재현 방법

```bash
pip install -r requirements.txt
python data/download.py        # Zenodo에서 데이터 수령 (재배포하지 않음 — 원출처 CC-BY)
jupyter notebook notebooks/    # 01 재현 → 02 지속 분포
streamlit run app/demo.py      # RAG-lite 데모
```

## 인용 논문

| 역할 | 논문 |
|---|---|
| 앵커(재현) | Somani et al., Communications Medicine 2024 |
| 품질 기준 | Sehgal et al., Nature Health 2026 ([프리프린트](https://arxiv.org/abs/2603.12341)) |
| 추출 수법 | Duan et al., AMIA 2025 ([arXiv](https://arxiv.org/abs/2504.04346)) |
| 한국 선례 | Lee group, Scientific Reports 2023 · JMIR Med Inform 2024 (네이버+KAERS 대조) |
| 중단율 근거 | Van Laren et al., 2026 (Kaiser EHR) |

## 한계

- 데이터는 2023-10까지의 영어 Reddit — 위고비 국내 출시 이후 한국어 담론은 미포함(다음 단계)
- 활동 지속 ≠ 복약 지속 — 대리지표
- LLM 라벨링은 원논문 모델 대체 — 수치의 완전 재현이 아니라 경향 재현

## License

코드 MIT · 데이터는 원출처(Zenodo, CC-BY 4.0) 인용 — 본 저장소는 데이터를 재배포하지 않습니다.

# %% [markdown]
# # 01 — Somani 재현 (축소판)
# 목표: 사전계산 2D 임베딩 로드 → 군집 → 토픽 지도 1장 → 원논문과 정성 대조.
# 축소 명시: LLM 토픽 라벨링·168토픽 정량 대조는 범위에서 제외(문서 II장에 명시).
# 실행: VS Code에서 셀(# %%) 단위 실행. 경로만 본인 환경에 맞출 것.

# %% 0. 경로 설정 — 본인 다운로드 위치로 수정
from pathlib import Path

DATA_DIR = Path(r"C:\data\glp1")  # TODO: Zenodo 파일을 둔 폴더로 수정
RAW_PICKLE = DATA_DIR / "glp1_db_20231019_edited.pickle"        # 212.8MB 원데이터
EMB_PICKLE = DATA_DIR / "glp1_embeddings_bgebase_2d.pickle"     # 3.1MB 사전계산 2D 임베딩

print("raw exists:", RAW_PICKLE.exists())
print("emb exists:", EMB_PICKLE.exists())

# %% 1. 열어서 정체 확인 (스키마를 모르는 상태 — 여기서 멈추면 출력 결과를 상담사에게)
import pandas as pd
import pickle
import numpy as np

print("pandas:", pd.__version__)

def load_any(p):
    try:
        return pd.read_pickle(p)
    except Exception as e:
        print("pd.read_pickle 실패:", type(e).__name__, e)
        with open(p, "rb") as f:
            return pickle.load(f)

emb = load_any(EMB_PICKLE)
print("EMB type:", type(emb))
if isinstance(emb, pd.DataFrame):
    print("shape:", emb.shape)
    print("columns:", list(emb.columns)[:30])
    print(emb.head(3))
elif isinstance(emb, dict):
    print("keys:", list(emb.keys())[:30])
else:
    print(repr(emb)[:500])

# %% 2. 원데이터도 같은 방식으로 확인 (메모리 주의 — 212MB)
raw = load_any(RAW_PICKLE)
print("RAW type:", type(raw))
if isinstance(raw, pd.DataFrame):
    print("shape:", raw.shape)
    print("columns:", list(raw.columns)[:40])
    print(raw.head(2))

# %% 3. 2D 좌표 추출 — 아래 컬럼명은 1번 셀 출력을 보고 맞출 것
# 예상: emb 가 (N, 2) 배열이거나, DataFrame 에 x/y 또는 0/1 컬럼
if isinstance(emb, np.ndarray):
    XY = emb
elif isinstance(emb, pd.DataFrame):
    num_cols = emb.select_dtypes("number").columns[:2]
    XY = emb[num_cols].to_numpy()
    print("사용 컬럼:", list(num_cols))
else:
    raise SystemExit("emb 구조를 1번 셀 출력으로 확인 후 이 셀을 수정")
print("XY shape:", XY.shape)

# %% 4. 군집 — HDBSCAN 우선, 실패 시 KMeans 폴백
N_FALLBACK_CLUSTERS = 33  # 원논문 33개 토픽 그룹에 맞춘 폴백

labels = None
try:
    import hdbscan
    clu = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=25)
    labels = clu.fit_predict(XY)
    method = "HDBSCAN"
except Exception as e:
    print("hdbscan 불가 → KMeans 폴백:", e)
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=N_FALLBACK_CLUSTERS, random_state=42, n_init=10).fit_predict(XY)
    method = f"KMeans(k={N_FALLBACK_CLUSTERS})"

import collections
cnt = collections.Counter(labels)
n_clusters = len([k for k in cnt if k != -1])
print(f"{method} → 군집 {n_clusters}개, 노이즈 {cnt.get(-1, 0)}건")
print("상위 10개 군집 크기:", cnt.most_common(10))

# %% 5. 토픽 지도 그림 — figures/topic_map.png 저장 (문서 II장 결과 그림)
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures" if "__file__" in dir() else Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
mask = labels != -1
ax.scatter(XY[~mask, 0], XY[~mask, 1], s=0.3, c="lightgray", alpha=0.3)
ax.scatter(XY[mask, 0], XY[mask, 1], s=0.3, c=labels[mask], cmap="tab20", alpha=0.5)
ax.set_title(f"GLP-1 Reddit discourse map — {method}, {n_clusters} clusters (reproduction)")
ax.set_xticks([]); ax.set_yticks([])
x_lo, x_hi = np.percentile(XY[:, 0], [0.5, 99.5])
y_lo, y_hi = np.percentile(XY[:, 1], [0.5, 99.5])
pad_x, pad_y = (x_hi - x_lo) * 0.05, (y_hi - y_lo) * 0.05
ax.set_xlim(x_lo - pad_x, x_hi + pad_x); ax.set_ylim(y_lo - pad_y, y_hi + pad_y)
out = FIG_DIR / "topic_map.png"
fig.savefig(out, bbox_inches="tight")
print("저장:", out)

# %% 6. 대표 군집 정성 대조 — 상위 군집별 샘플 텍스트 확인 → 원논문 33그룹과 비교 메모
# raw 와 emb 의 행 정렬이 같다는 가정 — 다르면 공통 id 컬럼으로 join 필요 (1·2번 셀 출력 참고)
TEXT_COL = "body"
if isinstance(raw, pd.DataFrame) and TEXT_COL and TEXT_COL in raw.columns and len(raw) == len(XY):
    raw = raw.assign(cluster=labels)
    for cid, _ in cnt.most_common(8):
        if cid == -1:
            continue
        sample = raw.loc[raw.cluster == cid, TEXT_COL].dropna().head(3)
        print(f"\n=== 군집 {cid} ({cnt[cid]}건) ===")
        for s in sample:
            print("-", str(s)[:150].replace("\n", " "))
else:
    print("TEXT_COL 지정 또는 행 정렬 확인 후 실행 — 1·2번 셀 출력을 보고 결정")

# %% 7. 대조 메모 (문서 II장 3절에 옮길 것)
# - 원논문: 168 토픽 → 33 그룹 (BGE + UMAP + HDBSCAN/KMeans + Llama2 라벨)
# - 본 재현: 사전계산 2D 임베딩 + {method} → {n_clusters} 군집
# - 차이와 원인: LLM 라벨링 미수행(축소), 2D 좌표 위 군집이므로 원논문(고차원 군집)과 수치가 다를 수 있음 — 정성 대조로 갈음

# %% [markdown]
# # 02 — 저자별 활동 지속 분포 (신규 산출 정리)
# 이미 산출한 결과(중앙값 65일 · 1년 잔존 8.2%)를 재현 가능한 스크립트로 공식화.
# ⚠️ 해석 경계: 게시 "활동"의 지속이지 "복약" 지속이 아님 — 대리지표. 복약 중단율은 EHR 연구(Van Laren 2026, 180일 31.7%) 인용.

# %% 0. 경로 — 01과 동일
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(r"C:\data\glp1")  # TODO: 본인 경로
RAW_PICKLE = DATA_DIR / "glp1_db_20231019_edited.pickle"

raw = pd.read_pickle(RAW_PICKLE)
print(raw.shape)
print(list(raw.columns)[:40])

# %% 1. 컬럼 매핑 — 출력 보고 수정
AUTHOR_COL = "author"          # TODO: 저자 컬럼명
TIME_COL = "created_utc"       # TODO: 시각 컬럼명 (epoch 초 또는 datetime)

t = pd.to_numeric(raw[TIME_COL], errors="coerce")
ts = pd.to_datetime(t, unit="s", errors="coerce")
df = pd.DataFrame({"author": raw[AUTHOR_COL].astype(str), "ts": ts}).dropna()
print("날짜 범위 확인:", df.ts.min(), "~", df.ts.max())  # 2019~2023이 나와야 정상
df = df[~df.author.isin(["[deleted]", "AutoModerator", "nan"])]
print("행:", len(df), "· 저자:", df.author.nunique())

# %% 2. 저자별 활동 기간 (첫 글 ~ 마지막 글)
g = df.groupby("author")["ts"].agg(first="min", last="max", n="count")
g["span_days"] = (g["last"] - g["first"]).dt.days
multi = g[g.n >= 2]  # 글 1건뿐인 저자는 기간 정의 불가 — 별도 보고
print(f"전체 저자 {len(g):,} · 2건 이상 {len(multi):,} ({len(multi)/len(g):.1%})")
print(f"활동 기간 중앙값: {multi.span_days.median():.0f}일")
for d in (30, 65, 90, 180, 365):
    print(f"  {d:>3}일 이상 지속: {(multi.span_days >= d).mean():.1%}")

# %% 3. 잔존 곡선 (Kaplan-Meier 스타일 단순판) + 그림 저장
import matplotlib.pyplot as plt

days = np.arange(0, 366)
surv = [(multi.span_days >= d).mean() for d in days]

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures" if "__file__" in dir() else Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.plot(days, surv, lw=2)
med = multi.span_days.median()
ax.axvline(med, ls="--", lw=1, color="gray")
ax.annotate(f"median {med:.0f}d", (med, 0.5), textcoords="offset points", xytext=(6, 6))
ax.axvline(365, ls=":", lw=1, color="gray")
ax.set_xlabel("days since first post"); ax.set_ylabel("share of authors still active")
ax.set_title("GLP-1 subreddit author activity persistence (proxy — not medication adherence)")
ax.set_ylim(0, 1)
out = FIG_DIR / "activity_survival.png"
fig.savefig(out, bbox_inches="tight")
print("저장:", out, f"· 1년 잔존 {surv[365]:.1%}")

# %% 4. 문서 기입용 요약 (II장 결과 · III장 대시보드 근거)
print(f"""
[문서 기입용]
- 분석 대상 저자: {len(g):,}명 (2건 이상 {len(multi):,}명)
- 활동 기간 중앙값: {multi.span_days.median():.0f}일 / 1년 지속: {(multi.span_days >= 365).mean():.1%}
- 한계: 게시 활동의 대리지표 — 복약 지속 아님. 단일 게시 저자 {1 - len(multi)/len(g):.1%} 제외 기준 명시.
""")

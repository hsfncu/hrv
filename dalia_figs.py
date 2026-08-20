"""Figures 2 and 3 for Paper 1."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut

OUT = Path(__file__).parent / "dalia_out"
df = pd.read_csv(OUT / "windows.csv")
CARD = ["ecg_hr", "ecg_sdnn", "ecg_rmssd", "ecg_pnn50"]
ALL = CARD + ["motion", "motion_sd", "temp"]
df = df.dropna(subset=CARD + ["eda_mean", "temp", "motion"]).copy()
df["eda_within"] = df.groupby("subject").eda_mean.transform(
    lambda s: (s - s.mean()) / s.std())

# ── Figure 2: Bland-Altman, wrist PPG vs chest ECG RMSSD ──────────
both = df.dropna(subset=["ppg_rmssd", "ecg_rmssd"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))

for a, m, unit in [(ax[0], "hr", "bpm"), (ax[1], "rmssd", "ms")]:
    x = (both[f"ecg_{m}"] + both[f"ppg_{m}"]) / 2
    d = both[f"ppg_{m}"] - both[f"ecg_{m}"]
    bias, sd = d.mean(), d.std()
    a.scatter(x, d, s=4, alpha=.15, color="#2C7899", edgecolors="none")
    a.axhline(bias, color="#99202F", lw=1.4, label=f"bias {bias:+.2f}")
    a.axhline(bias + 1.96 * sd, color="#99202F", ls="--", lw=.9,
              label=f"±1.96 SD ({1.96*sd:.1f})")
    a.axhline(bias - 1.96 * sd, color="#99202F", ls="--", lw=.9)
    a.axhline(0, color="#A7B5C1", lw=.8)
    a.set(xlabel=f"mean of ECG and PPG ({unit})",
          ylabel=f"PPG − ECG ({unit})",
          title=f"{m.upper()}   r = {both[f'ecg_{m}'].corr(both[f'ppg_{m}']):+.3f}")
    a.legend(fontsize=8, loc="upper left")
    a.grid(alpha=.25, lw=.5)
fig.suptitle("Wrist PPG against chest ECG "
             f"({len(both):,} windows, {100*len(both)/len(df):.1f}% coverage)",
             y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig2_bland_altman.png", dpi=200, bbox_inches="tight")
print("wrote fig2_bland_altman.png")

# ── Figure 3: activity-stratified bias ────────────────────────────
X, y, g = df[ALL].values, df.eda_within.values, df.subject.values
logo = LeaveOneGroupOut()
yt, yp, ac = [], [], []
for tr, te in logo.split(X, y, g):
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                              random_state=0, n_jobs=-1)
    m.fit(X[tr], y[tr])
    yt.append(y[te]); yp.append(m.predict(X[te])); ac.append(df.activity.values[te])
yt, yp, ac = map(np.concatenate, (yt, yp, ac))

rows = []
for a_ in df.activity.unique():
    mk = ac == a_
    if mk.sum() < 100:
        continue
    sub = df[df.activity == a_]
    rows.append(dict(activity=a_, n=int(mk.sum()),
                     motion=sub.motion.mean(), hr=sub.ecg_hr.mean(),
                     true=yt[mk].mean(), pred=yp[mk].mean(),
                     bias=yp[mk].mean() - yt[mk].mean()))
r = pd.DataFrame(rows).sort_values("motion")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
xx = np.arange(len(r))
ax[0].barh(xx - .2, r.true, .4, label="measured EDA", color="#17516E")
ax[0].barh(xx + .2, r.pred, .4, label="predicted", color="#C08A2E")
ax[0].axvline(0, color="#A7B5C1", lw=.8)
ax[0].set(yticks=xx, xlabel="EDA (within-subject SD)",
          title="Measured vs predicted by activity")
ax[0].set_yticklabels(r.activity, fontsize=8)
ax[0].legend(fontsize=8); ax[0].grid(alpha=.25, lw=.5, axis="x")

cols = ["#99202F" if b > 0 else "#1D6746" for b in r.bias]
ax[1].barh(xx, r.bias, color=cols)
ax[1].axvline(0, color="#A7B5C1", lw=.8)
ax[1].set(yticks=xx, xlabel="prediction bias (pred − true, SD)",
          title="Bias, ordered by motion (low → high)")
ax[1].set_yticklabels([f"{a}  ({m:.4f} g)" for a, m in zip(r.activity, r.motion)],
                      fontsize=7)
ax[1].grid(alpha=.25, lw=.5, axis="x")
fig.tight_layout()
fig.savefig(OUT / "fig3_activity_bias.png", dpi=200, bbox_inches="tight")
print("wrote fig3_activity_bias.png")
print(r.round(4).to_string(index=False))

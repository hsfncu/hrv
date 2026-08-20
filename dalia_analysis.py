"""
PPG-DaLiA — core analysis for Paper 1.

Can wrist EDA be estimated from cardiac + motion + temperature signals?

HRV here comes from chest-ECG R-peaks (gold standard, 98.8% window coverage).
Wrist-PPG HRV is reported separately because it carries a +61 ms RMSSD bias
and only 38% coverage -- that gap is a result, not a nuisance.

Every model is scored with leave-one-subject-out. Random K-fold is shown
alongside purely to document how much it flatters the number: on the earlier
synthetic dataset K-fold said R^2 = 0.70 while the honest cross-condition
figure was -37.

Two targets, kept separate on purpose:
  RAW     absolute EDA in uS. Between-subject SCL spans ~9x here, so this
          asks the hard question.
  WITHIN  EDA z-scored inside each subject. Asks whether a person's own
          arousal changes can be tracked, which is what a wearable needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUT = Path(__file__).parent / "dalia_out"
R = "=" * 78
def head(t): print(f"\n{R}\n{t}\n{R}")

CARD = ["ecg_hr", "ecg_sdnn", "ecg_rmssd", "ecg_pnn50"]
MOT = ["motion", "motion_sd"]

df = pd.read_csv(OUT / "windows.csv")
df = df.dropna(subset=CARD + ["eda_mean", "temp", "motion"]).copy()
df["eda_within"] = df.groupby("subject").eda_mean.transform(
    lambda s: (s - s.mean()) / s.std())

FEATSETS = {
    "HR":              ["ecg_hr"],
    "HR+HRV":          CARD,
    "HR+HRV+MOT":      CARD + MOT,
    "HR+HRV+MOT+TEMP": CARD + MOT + ["temp"],
}
ALL = FEATSETS["HR+HRV+MOT+TEMP"]
groups = df.subject.values
logo = LeaveOneGroupOut()

print(f"{len(df):,} windows | {df.subject.nunique()} subjects | "
      f"{len(df)/df.subject.nunique():.0f} per subject")


# ══════════════════════════════════════════════════════════════════
head("1 — descriptives")
for c in ALL + ["eda_mean"]:
    print(f"{c:<12}{df[c].mean():>10.2f}{df[c].std():>9.2f}"
          f"{df[c].min():>9.2f}{df[c].max():>9.2f}")
g = df.groupby("subject").eda_mean.agg(["mean", "std"])
print(f"\nbetween-subject mean EDA: {g['mean'].min():.2f} - "
      f"{g['mean'].max():.2f} uS  ({g['mean'].max()/g['mean'].min():.1f}x)")


# ══════════════════════════════════════════════════════════════════
head("2 — pooled vs within-subject correlation with EDA")
print("The synthetic dataset failed here: pooled +0.759, within +0.07.\n")
print(f"{'feature':<12}{'pooled':>10}{'within mean':>14}{'range':>22}")
for c in ALL:
    p = df[c].corr(df.eda_mean)
    w = df.groupby("subject").apply(
        lambda x, c=c: x[c].corr(x.eda_mean), include_groups=False)
    print(f"{c:<12}{p:>+10.3f}{w.mean():>+14.3f}   [{w.min():+.3f}, {w.max():+.3f}]")


# ══════════════════════════════════════════════════════════════════
def evaluate(X, y, groups, fn, cv):
    yt, yp, gs = [], [], []
    for tr, te in cv.split(X, y, groups):
        m = fn(); m.fit(X[tr], y[tr])
        yt.append(y[te]); yp.append(m.predict(X[te])); gs.append(groups[te])
    yt, yp, gs = map(np.concatenate, (yt, yp, gs))
    per = [r2_score(yt[gs == s], yp[gs == s]) for s in np.unique(gs)]
    return dict(r2=r2_score(yt, yp), mae=mean_absolute_error(yt, yp),
                per_sub=per, sub_mean=float(np.mean(per)),
                sub_med=float(np.median(per)), yt=yt, yp=yp, gs=gs)


def poly(d):
    return lambda: make_pipeline(PolynomialFeatures(d, include_bias=False),
                                 StandardScaler(), Ridge(alpha=1.0))


def rf():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                 random_state=0, n_jobs=-1)


head("3 — LOSO regression")
for target, tname in [("eda_mean", "RAW (absolute uS)"),
                      ("eda_within", "WITHIN (z-scored per subject)")]:
    print(f"\n--- target: {tname} ---")
    y = df[target].values
    print(f"{'features':<20}{'model':<9}{'LOSO R2':>10}{'MAE':>9}"
          f"{'per-subj':>11}{'5-fold':>10}")
    for name, feats in FEATSETS.items():
        X = df[feats].values
        for mn, fn in [("ridge2", poly(2)), ("RF", rf)]:
            a = evaluate(X, y, groups, fn, logo)
            b = evaluate(X, y, groups, fn, KFold(5, shuffle=True, random_state=0))
            print(f"{name:<20}{mn:<9}{a['r2']:>10.3f}{a['mae']:>9.3f}"
                  f"{a['sub_mean']:>11.3f}{b['r2']:>10.3f}")


# ══════════════════════════════════════════════════════════════════
head("4 — per-subject generalisation (RF, WITHIN target)")
X, y = df[ALL].values, df.eda_within.values
res = evaluate(X, y, groups, rf, logo)
subs = np.unique(groups)
print(f"{'subject':<9}{'LOSO R2':>10}{'mean EDA':>11}{'age':>6}{'n':>8}")
for s, r2 in sorted(zip(subs, res["per_sub"]), key=lambda t: -t[1]):
    sub = df[df.subject == s]
    print(f"{s:<9}{r2:>10.3f}{sub.eda_mean.mean():>11.2f}"
          f"{sub.age.iloc[0]:>6.0f}{len(sub):>8,}")
print(f"\nmean {res['sub_mean']:.3f}   median {res['sub_med']:.3f}   "
      f"positive in {sum(r > 0 for r in res['per_sub'])}/{len(subs)} subjects")


# ══════════════════════════════════════════════════════════════════
head("5 — physical load vs arousal: where does the model go wrong?")
yt, yp, acts = res["yt"], res["yp"], None
acts = np.concatenate([df.activity.values[te] for _, te in logo.split(X, y, groups)])
print(f"{'activity':<15}{'n':>7}{'HR':>8}{'motion':>9}{'true':>8}"
      f"{'pred':>8}{'bias':>8}{'R2':>8}")
for a in df.activity.value_counts().index:
    m = acts == a
    if m.sum() < 100:
        continue
    sub = df[df.activity == a]
    print(f"{a:<15}{m.sum():>7,}{sub.ecg_hr.mean():>8.1f}"
          f"{sub.motion.mean():>9.4f}{yt[m].mean():>8.3f}{yp[m].mean():>8.3f}"
          f"{yp[m].mean()-yt[m].mean():>+8.3f}{r2_score(yt[m], yp[m]):>8.3f}")


# ══════════════════════════════════════════════════════════════════
head("6 — permutation importance (held-out)")
tr, te = next(GroupKFold(n_splits=5).split(X, y, groups))
m = rf(); m.fit(X[tr], y[tr])
pi = permutation_importance(m, X[te], y[te], n_repeats=10,
                            random_state=0, n_jobs=-1)
for i in np.argsort(-pi.importances_mean):
    print(f"{ALL[i]:<14}{pi.importances_mean[i]:>10.4f}"
          f" +- {pi.importances_std[i]:.4f}")


# ══════════════════════════════════════════════════════════════════
head("7 — wrist-PPG HRV vs chest-ECG HRV")
both = df.dropna(subset=["ppg_hr", "ppg_rmssd"])
print(f"{len(both):,} windows have both ({100*len(both)/len(df):.1f}%)\n")
print(f"{'metric':<9}{'r':>9}{'bias':>10}{'MAE':>9}")
for mm in ("hr", "sdnn", "rmssd", "pnn50"):
    r = both[f"ecg_{mm}"].corr(both[f"ppg_{mm}"])
    bias = (both[f"ppg_{mm}"] - both[f"ecg_{mm}"]).mean()
    mae = (both[f"ppg_{mm}"] - both[f"ecg_{mm}"]).abs().mean()
    print(f"{mm:<9}{r:>+9.3f}{bias:>+10.2f}{mae:>9.2f}")

print("\nSame model, but HRV taken from wrist PPG instead of chest ECG:")
pf = ["ppg_hr", "ppg_sdnn", "ppg_rmssd", "ppg_pnn50"] + MOT + ["temp"]
sub = df.dropna(subset=pf)
if sub.subject.nunique() > 3:
    r_ppg = evaluate(sub[pf].values, sub.eda_within.values,
                     sub.subject.values, rf, LeaveOneGroupOut())
    r_ecg = evaluate(sub[ALL].values, sub.eda_within.values,
                     sub.subject.values, rf, LeaveOneGroupOut())
    print(f"  chest-ECG HRV : LOSO R2 {r_ecg['r2']:+.3f}")
    print(f"  wrist-PPG HRV : LOSO R2 {r_ppg['r2']:+.3f}")


# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))

order = g["mean"].sort_values().index
ax[0].bar(range(len(order)), g.loc[order, "mean"], yerr=g.loc[order, "std"],
          color="#2C7899", ecolor="#A7B5C1", capsize=2)
ax[0].set(xticks=range(len(order)), xlabel="subject", ylabel="EDA (µS)",
          title=f"Between-subject SCL spans {g['mean'].max()/g['mean'].min():.0f}×")
ax[0].set_xticklabels(order, rotation=60, fontsize=7)
ax[0].grid(alpha=.25, lw=.5, axis="y")

pool = [df[c].corr(df.eda_mean) for c in ALL]
with_ = [df.groupby("subject").apply(
    lambda x, c=c: x[c].corr(x.eda_mean), include_groups=False).mean() for c in ALL]
xx = np.arange(len(ALL))
ax[1].bar(xx - .2, pool, .4, label="pooled", color="#99202F")
ax[1].bar(xx + .2, with_, .4, label="within-subject", color="#2C7899")
ax[1].axhline(0, color="#A7B5C1", lw=.8)
ax[1].set(xticks=xx, ylabel="r with EDA", title="Pooled vs within-subject r")
ax[1].set_xticklabels([c.replace("ecg_", "") for c in ALL], rotation=45,
                      ha="right", fontsize=8)
ax[1].legend(fontsize=8); ax[1].grid(alpha=.25, lw=.5, axis="y")

ps = sorted(zip(subs, res["per_sub"]), key=lambda t: t[1])
cols = ["#99202F" if p[1] < 0 else "#1D6746" for p in ps]
ax[2].barh([p[0] for p in ps], [p[1] for p in ps], color=cols)
ax[2].axvline(0, color="#A7B5C1", lw=.8)
ax[2].set(xlabel="LOSO $R^2$", title="Per-subject generalisation")
ax[2].tick_params(labelsize=7); ax[2].grid(alpha=.25, lw=.5, axis="x")

fig.tight_layout()
fig.savefig(OUT / "fig_main.png", dpi=200)
print(f"\nwrote {OUT/'fig_main.png'}")

pd.DataFrame(dict(subject=subs, loso_r2=res["per_sub"])).to_csv(
    OUT / "per_subject_r2.csv", index=False)
print("DONE.")

"""
Does the PPG-DaLiA conclusion hold on WESAD?

WESAD is the harder test in one respect and the easier one in another. Easier:
arousal is experimentally manipulated by the Trier Social Stress Test, so there
is real sympathetic variation to predict rather than whatever incidental
arousal a free-living day contains. Harder for the claim: if EDA really is
recoverable from cardiac signals, this is where it should show.

WESAD is also, by construction, exactly the design Section 5.2 of the paper
warns about -- four discrete conditions -- so any apparent success has to be
checked against the possibility that the model is recognising the condition
rather than estimating the signal. Three analyses separate those:

  * pooled vs within-condition correlations
  * leave-one-subject-out, as in the main paper
  * leave-one-condition-out, which asks whether anything transfers to a
    protocol phase the model has never seen

WESAD additionally carries chest EDA, which PPG-DaLiA does not. That allows
the question the main paper cannot ask: how well do two directly measured EDA
channels agree with each other? It bounds what any indirect estimate could
achieve.

Reads ./wesad_out/windows.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUT = Path(__file__).parent / "wesad_out"
df = pd.read_csv(OUT / "windows.csv")

CARD = ["ecg_hr", "ecg_sdnn", "ecg_rmssd", "ecg_pnn50"]
SETS = {
    "HR": ["ecg_hr"],
    "HR+HRV": CARD,
    "HR+HRV+MOT": CARD + ["motion", "motion_sd"],
    "HR+HRV+MOT+TEMP": CARD + ["motion", "motion_sd", "temp"],
}
df = df.dropna(subset=CARD + ["wrist_eda", "chest_eda", "temp", "motion"]).copy()
df["eda_within"] = df.groupby("subject").wrist_eda.transform(
    lambda s: (s - s.mean()) / s.std())
grp = df.subject.values


def head(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


head("0 - dataset")
print(f"{len(df):,} windows, {df.subject.nunique()} subjects")
print(df.condition.value_counts().to_string())

# ── 1. pooled vs within-condition vs within-subject ──────────────
head("1 - correlation with wrist EDA: pooled vs within-condition")
print(f"{'feature':<14}{'pooled r':>10}{'within-cond':>13}{'within-subj':>13}")
print("-" * 50)
for f in ["ecg_hr", "ecg_sdnn", "ecg_rmssd", "ecg_pnn50",
          "motion", "motion_sd", "temp"]:
    pooled = df[f].corr(df.wrist_eda)
    wc = df.groupby("condition").apply(
        lambda g: g[f].corr(g.wrist_eda), include_groups=False).mean()
    ws = df.groupby("subject").apply(
        lambda g: g[f].corr(g.wrist_eda), include_groups=False).mean()
    print(f"{f:<14}{pooled:>+10.3f}{wc:>+13.3f}{ws:>+13.3f}")
print("\nA pooled value much larger than the within-condition value means the "
      "association\nis between condition means, not within them.")

# ── 2. models ────────────────────────────────────────────────────
ridge2 = lambda: make_pipeline(PolynomialFeatures(2, include_bias=False),
                               StandardScaler(), Ridge(alpha=1.0))
rf = lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                   random_state=0, n_jobs=-1)


def ev(X, y, groups, cv, fn):
    yt, yp, gs = [], [], []
    for tr, te in cv.split(X, y, groups):
        m = fn(); m.fit(X[tr], y[tr])
        yt.append(y[te]); yp.append(m.predict(X[te])); gs.append(groups[te])
    yt, yp, gs = map(np.concatenate, (yt, yp, gs))
    per = [r2_score(yt[gs == s], yp[gs == s]) for s in np.unique(gs)]
    return r2_score(yt, yp), float(np.mean(np.abs(yt - yp))), per


head("2 - LOSO vs random 5-fold (target: within-subject standardised EDA)")
print(f"{'features':<18}{'model':<8}{'LOSO R2':>9}{'MAE':>8}"
      f"{'5-fold R2':>11}{'inflation':>11}")
print("-" * 65)
y = df.eda_within.values
best = (-9, None)
for name, cols in SETS.items():
    X = df[cols].values
    for mname, fn in (("ridge2", ridge2), ("RF", rf)):
        lo, mae, per = ev(X, y, grp, LeaveOneGroupOut(), fn)
        kf, _, _ = ev(X, y, grp, KFold(5, shuffle=True, random_state=0), fn)
        mark = " *" if lo > best[0] else ""
        if lo > best[0]:
            best = (lo, (name, mname, cols, per))
        print(f"{name:<18}{mname:<8}{lo:>+9.3f}{mae:>8.3f}"
              f"{kf:>+11.3f}{kf-lo:>+11.3f}{mark}")

head("3 - the same, target: raw wrist EDA in microsiemens")
print(f"{'features':<18}{'model':<8}{'LOSO R2':>9}{'MAE':>8}"
      f"{'5-fold R2':>11}{'inflation':>11}")
print("-" * 65)
yr = df.wrist_eda.values
for name, cols in SETS.items():
    X = df[cols].values
    for mname, fn in (("ridge2", ridge2), ("RF", rf)):
        lo, mae, _ = ev(X, yr, grp, LeaveOneGroupOut(), fn)
        kf, _, _ = ev(X, yr, grp, KFold(5, shuffle=True, random_state=0), fn)
        print(f"{name:<18}{mname:<8}{lo:>+9.3f}{mae:>8.3f}"
              f"{kf:>+11.3f}{kf-lo:>+11.3f}")

# ── 4. baseline every model has to beat ──────────────────────────
head("4 - trivial baselines")
print(f"predicting each subject's own mean (within target): "
      f"MAE {np.abs(df.eda_within).mean():.3f} SD")
lo, mae, per = ev(df[best[1][2]].values, y, grp, LeaveOneGroupOut(),
                  ridge2 if best[1][1] == "ridge2" else rf)
print(f"best model ({best[1][0]}, {best[1][1]}):            "
      f"MAE {mae:.3f} SD   -> "
      f"{100*(np.abs(df.eda_within).mean()-mae)/np.abs(df.eda_within).mean():.1f}% better")
print(f"\nper-subject R2 positive in {sum(p > 0 for p in best[1][3])}/"
      f"{len(best[1][3])}, mean {np.mean(best[1][3]):+.3f}, "
      f"median {np.median(best[1][3]):+.3f}")

# ── 5. leave-one-condition-out ───────────────────────────────────
head("5 - leave-one-condition-out (does anything transfer to an unseen phase?)")
cg = df.condition.values
X = df[SETS["HR+HRV+MOT+TEMP"]].values
for cv_name, fn in (("ridge2", ridge2), ("RF", rf)):
    lo, mae, per = ev(X, y, cg, LeaveOneGroupOut(), fn)
    print(f"  {cv_name:<8} pooled R2 {lo:+.3f}   per-condition "
          f"{dict(zip(sorted(np.unique(cg)), np.round(per, 3)))}")

# ── 6. permutation importance ────────────────────────────────────
head("6 - permutation importance (held-out, grouped by subject)")
cols = SETS["HR+HRV+MOT+TEMP"]
X = df[cols].values
tr, te = next(GroupKFold(5).split(X, y, grp))
m = rf(); m.fit(X[tr], y[tr])
pi = permutation_importance(m, X[te], y[te], n_repeats=10, random_state=0,
                            n_jobs=-1)
for i in np.argsort(pi.importances_mean)[::-1]:
    print(f"  {cols[i]:<12}{pi.importances_mean[i]:>8.4f} "
          f"+- {pi.importances_std[i]:.4f}")

# ── 7. the question PPG-DaLiA cannot ask ─────────────────────────
head("7 - wrist EDA vs chest EDA: two direct measurements of the same thing")
r_pool = df.wrist_eda.corr(df.chest_eda)
r_ws = df.groupby("subject").apply(
    lambda g: g.wrist_eda.corr(g.chest_eda), include_groups=False)
print(f"pooled r = {r_pool:+.3f}")
print(f"within-subject r: mean {r_ws.mean():+.3f}, "
      f"median {r_ws.median():+.3f}, "
      f"range [{r_ws.min():+.3f}, {r_ws.max():+.3f}]")
print(f"positive in {int((r_ws > 0).sum())}/{len(r_ws)} subjects")
print("\nper condition:")
for c, g in df.groupby("condition"):
    print(f"  {c:<12}{g.wrist_eda.corr(g.chest_eda):+.3f}  (n={len(g)})")
print("\nThis is a practical benchmark, not a formal bound: if two electrodes "
      "both measuring skin\nconductance agree only this well, an estimate from "
      "cardiac signals is unlikely to do better.")

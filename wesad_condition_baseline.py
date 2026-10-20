"""Condition recognition versus EDA reconstruction in WESAD (Section 4.7.2).

Uses wesad_out/windows.csv and the same model and target as wesad_analysis.py:
condition-only baseline, within-condition performance, and the model refitted
after removing condition-level differences.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

df = pd.read_csv(Path(sys.argv[1]) if len(sys.argv) > 1
                 else Path(__file__).parent / "wesad_out" / "windows.csv")
CARD = ["ecg_hr", "ecg_sdnn", "ecg_rmssd", "ecg_pnn50"]
FEATS = CARD + ["motion", "motion_sd"]           # best WESAD set (HR+HRV+MOT)
df = df.dropna(subset=CARD + ["wrist_eda", "chest_eda", "temp", "motion"]).copy()
df["y"] = df.groupby("subject").wrist_eda.transform(lambda s: (s - s.mean()) / s.std())
ridge2 = lambda: make_pipeline(PolynomialFeatures(2, include_bias=False),
                               StandardScaler(), Ridge(alpha=1.0))
logo, g = LeaveOneGroupOut(), df.subject.values
conds = sorted(df.condition.unique())
onehot = pd.get_dummies(df.condition).astype(float).values


def loso(X, y, fn=ridge2):
    p = np.empty_like(y)
    for tr, te in logo.split(X, y, g):
        m = fn(); m.fit(X[tr], y[tr]); p[te] = m.predict(X[te])
    return p


def cond_only(y):
    p = np.empty_like(y)
    for tr, te in logo.split(y, y, g):
        mu = pd.Series(y[tr]).groupby(df.condition.values[tr]).mean()
        p[te] = mu.reindex(df.condition.values[te]).values
    return p


def report(name, y, p):
    per = [r2_score(y[g == s], p[g == s]) for s in np.unique(g)]
    print(f"{name:<44}R2 {r2_score(y, p):+.3f}  MAE {np.mean(np.abs(y-p)):.3f}"
          f"  per-subj>0 {sum(r > 0 for r in per)}/15")
    return p


y = df.y.values
X = df[FEATS].values
print(f"{len(df)} windows\n")
print("A. LOSO, target = within-subject z EDA")
p_mod = report("physiological model (ridge2, HR+HRV+MOT)", y, loso(X, y))
p_cnd = report("condition-only baseline (condition means)", y, cond_only(y))
report("condition labels + physiological features", y, loso(np.hstack([X, onehot]), y))

print("\nB. Within-condition performance of the physiological model")
for c in conds:
    k = df.condition.values == c
    print(f"  {c:<12} n={k.sum():>5}  R2 {r2_score(y[k], p_mod[k]):+.3f}"
          f"  r {np.corrcoef(y[k], p_mod[k])[0,1]:+.3f}")
# per subject x condition correlation, averaged
rs = [np.corrcoef(s.y, s.p)[0, 1] for _, s in
      df.assign(p=p_mod).groupby(["subject", "condition"]) if len(s) > 5 and s.y.std() > 0]
print(f"  mean within-(subject,condition) r = {np.nanmean(rs):+.3f} "
      f"(median {np.nanmedian(rs):+.3f}, {len(rs)} cells)")

print("\nC. After removing condition-level differences (centre target and features")
print("   on each subject x condition mean), refit, LOSO")
key = [df.subject, df.condition]
yc = (df.y - df.groupby(key).y.transform("mean")).values
Xc = (df[FEATS] - df.groupby(key)[FEATS].transform("mean")).values
report("physiological model on condition-centred data", yc, loso(Xc, yc))
print(f"  share of target variance between conditions (within subject): "
      f"{1 - yc.var() / y.var():.1%}")

# Wrist EDA recoverability — analysis code

Code and derived features for:

> Huang, S.-F. *Wrist Electrodermal Activity Is Not Recoverable from Cardiac and
> Motion Signals: A Leave-One-Subject-Out Analysis of 15 Subjects in Free-Living
> Conditions.*

The study asks whether wrist electrodermal activity (EDA) can be estimated from
signals a standard wrist wearable already provides — heart rate, heart rate
variability, motion and skin temperature. Under leave-one-subject-out (LOSO)
validation it cannot: best R² = 0.094, positive in 4 of 15 participants. The
same models under random 5-fold cross-validation report R² up to 0.670.

## Data

The analysis uses **PPG-DaLiA**, which is not redistributed here. Download it
from the UCI Machine Learning Repository:

<https://archive.ics.uci.edu/dataset/495/ppg+dalia>

Extract so that each subject's pickle sits at `PPG_FieldStudy/S{n}/S{n}.pkl`.
The archives are large (~1.4 GB per subject once extracted); `dalia_features.py`
processes one subject at a time and releases each before the next so peak disk
use stays near 1.5 GB rather than 22 GB.

HRV is computed from the **chest-ECG R-peak indices supplied with the dataset**
(700 Hz), not from wrist PPG. This is deliberate: it removes pulse-detection
error as a confound, so the negative result cannot be attributed to poor HRV
measurement. Wrist-PPG HRV is derived separately and compared against this
reference in Section 4.6 of the paper.

## Files

| File | Purpose |
|---|---|
| `dalia_features.py` | Reads the per-subject pickles; builds 60 s / 10 s-hop windows; RR-interval cleaning, HRV, EDA tonic/phasic split, gravity-removed motion, wrist-PPG peak detection |
| `dalia_analysis.py` | Descriptives, pooled vs within-subject correlations, LOSO and 5-fold regression over four nested feature sets, per-subject breakdown, permutation importance, PPG-vs-ECG comparison |
| `make_figures.py` | Publication figures (600 dpi, TrueType-embedded, colourblind-safe) |
| `dalia_figs.py` | Supporting exploratory plots |
| `extract_e4.py`, `probe_pkl.py` | Archive extraction and pickle inspection helpers |

## Precomputed outputs

`dalia_out/windows.csv` holds the 12,864 window-level feature rows derived from
PPG-DaLiA, so the modelling results can be reproduced without downloading and
reprocessing the raw dataset. `dalia_out/per_subject_r2.csv` holds the
per-participant LOSO scores plotted in Figure 1c.

## Reproducing the reported numbers

```bash
python -m venv venv
venv/Scripts/activate          # Windows;  source venv/bin/activate on Unix
pip install -r requirements.txt

python dalia_features.py       # only if regenerating windows.csv from raw data
python dalia_analysis.py       # all reported statistics
python make_figures.py         # Figures 1-3
```

Which output maps to what:

| Paper | Produced by |
|---|---|
| Table 1 (descriptives) | `dalia_analysis.py`, section 1 |
| Table 2 (pooled vs within-subject r) | `dalia_analysis.py`, section 2 |
| Table 3 (LOSO vs 5-fold) | `dalia_analysis.py`, section 3 |
| Table 4 (permutation importance) | `dalia_analysis.py`, section 5 |
| Table 5 (PPG vs ECG HRV) | `dalia_analysis.py`, section 6 |
| Figures 1–3 | `make_figures.py` |

## Two implementation details that change the answer

Both were bugs during development and are recorded here because either one
silently produces a wrong result:

1. **Sub-sample peak interpolation is not optional.** At 64 Hz one BVP sample
   spans 15.6 ms while daily-life RMSSD is 20–60 ms, so peak quantisation alone
   would be a large fraction of the measured quantity. `bvp_peak_times()`
   refines each peak parabolically.
2. **Accelerometer gravity must be removed.** Without a 0.3 Hz high-pass the
   acceleration magnitude is ≈1.0 g everywhere and carries no information about
   movement. An early version of this analysis used raw magnitude and the
   "motion" feature was in fact gravity.

## Environment

Python 3.14.7 with NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.0, scikit-learn 1.9.0,
matplotlib 3.11.1. Older versions are likely fine; the pins in
`requirements.txt` record what the reported numbers were produced with.

## Licence

MIT — see `LICENSE`. PPG-DaLiA itself is distributed under its own terms by the
UCI Machine Learning Repository; cite Reiss et al. (2019) if you use it.

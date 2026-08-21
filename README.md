# Bounds on wearable EDA estimation — analysis code

Code and derived features for:

> Huang, S.-F.; Huang, Y.-P. *Bounds on Estimating Electrodermal Activity from Wrist-Available
> Signals: Evidence from Two Public Datasets and a Paired-Measurement Ceiling.*

The study asks how much wrist electrodermal activity (EDA) can be recovered from
signals a standard wrist wearable already provides — heart rate, heart rate
variability, motion and skin temperature — and what sets the limit.

Under leave-one-subject-out validation, estimation reaches R² = 0.094 in
free-living recordings (PPG-DaLiA) and R² = 0.158 in laboratory recordings with
arousal experimentally induced (WESAD). The laboratory result is a positive
control: it shows the pipeline detects an association where one exists, so the
free-living null describes those data rather than the method. That association
nonetheless improves on predicting each participant's own mean by only 11.9%,
sits mostly between protocol conditions rather than within them, and transfers
to no unseen condition (leave-one-condition-out R² = −0.75).

A ceiling explains the shortfall. Wrist and chest EDA, recorded simultaneously
and both measured directly, correlate at only r = 0.485 — falling to 0.378 under
stress, and negative in one participant. No indirect estimate of skin conductance
at one site can exceed what direct measurement at another achieves.

## Data

Two public datasets, neither redistributed here.

**PPG-DaLiA** — free-living, 15 adults, eight daily activities.
<https://archive.ics.uci.edu/dataset/495/ppg+dalia>

**WESAD** — laboratory, 15 adults, Trier Social Stress Test protocol. Same
instrument pair (chest RespiBAN, wrist Empatica E4), and additionally carries
chest EDA, which is what makes the agreement ceiling in the paper measurable.
No registration or agreement is required; the licence permits scientific
non-commercial use with credit to the dataset authors.
<https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html>

WESAD serves as a positive control. A null result is only informative if the
same pipeline can detect an effect where one exists, and WESAD induces
sympathetic arousal deliberately rather than incidentally.

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
| `dalia_analysis.py` | Descriptives, pooled vs within-subject correlations, LOSO and 5-fold regression over four nested feature sets, per-subject breakdown, permutation importance |
| `ppg_coverage_sweep.py` | Sweeps the artifact-rejection threshold across all 15 subjects, producing the accuracy–coverage trade-off of Table 5 |
| `wesad_features.py` | WESAD windows: Pan–Tompkins R-peak detection (that dataset ships no peak indices), condition-pure windowing, and the three validation gates that must pass before any result is read |
| `wesad_analysis.py` | Pooled vs within-condition correlations, LOSO, leave-one-condition-out, permutation importance, and the wrist-vs-chest EDA agreement ceiling |
| `make_fig5.py` | Figure 5, the agreement ceiling and the importance reversal between datasets |
| `make_figures.py` | Figures 1–3 (600 dpi, TrueType-embedded, colourblind-safe) |
| `make_fig4.py` | Figure 4, the accuracy–coverage trade-off |
| `dalia_figs.py` | Supporting exploratory plots |
| `extract_e4.py`, `probe_pkl.py` | Archive extraction and pickle inspection helpers |

## Precomputed outputs

`dalia_out/windows.csv` holds the 12,864 window-level feature rows derived from
PPG-DaLiA, so the modelling results can be reproduced without downloading and
reprocessing the raw dataset. `dalia_out/per_subject_r2.csv` holds the
per-participant LOSO scores plotted in Figure 1c. `dalia_out/ppg_sweep.csv`
holds one row per window per rejection threshold (101,616 rows), and
`ppg_sweep_summary.csv` the aggregated curve.

## Why the wrist-PPG comparison is reported as a sweep

An earlier version of this analysis quoted a single agreement statistic for
wrist-PPG RMSSD: a bias of +61 ms at a 25% artifact-rejection threshold. That
number is reproducible from this code, but it should not have been presented
as a property of the device. Holding beat detection fixed and varying only the
rejection threshold moves the bias from +81.8 ms (49.5% of windows retained)
to +6.4 ms (2.7% retained).

What survives is the more useful result, and it is the one the paper reports:
agreement peaks at r = 0.585 and declines thereafter, the smallest achievable
bias is still about a quarter of the reference value, and the strictest
thresholds preferentially retain windows of genuinely lower variability. Run
`ppg_coverage_sweep.py` to reproduce the whole curve.

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
| Table 5 (accuracy–coverage sweep) | `ppg_coverage_sweep.py` |
| Figures 1–3 | `make_figures.py` |
| Figure 4 | `make_fig4.py` |

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


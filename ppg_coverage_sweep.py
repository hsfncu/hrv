"""
Accuracy-coverage trade-off for wrist-PPG heart rate variability.

A single agreement statistic for wrist-PPG RMSSD is not a property of the
device: it is a property of the artifact-rejection threshold chosen to produce
it. A permissive threshold retains most windows and reports a large bias; a
strict one reports near-zero bias on a small fraction of the recording. Quoting
either number alone is misleading.

This script sweeps the threshold across all 15 subjects and reports bias,
agreement and retention together, so the trade-off can be read off directly.

Beat detection is held fixed across the sweep (band-pass 0.7-3.5 Hz, 0.35 s
refractory, prominence 0.5 SD, parabolic sub-sample refinement) so that the
only thing varying is how aggressively intervals are rejected afterwards.

The ECG reference uses the dataset's own R-peak indices at 700 Hz with a fixed
25% local-median criterion throughout.

Writes ./dalia_out/ppg_sweep.csv (one row per window per threshold).
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, medfilt

DS = Path(__file__).parent.parent / "datasets"
INNER = DS / "outer" / "data.zip"
TMP = DS / "pkl_tmp"
OUT = Path(__file__).parent / "dalia_out"

FS_ECG, FS_BVP, FS_ACC = 700, 64, 32
WIN, HOP = 60.0, 10.0
MIN_BEATS, MAX_REJECT = 25, 0.20
ECG_TOL = 0.25

TOLS = [0.30, 0.25, 0.20, 0.15, 0.125, 0.10, 0.075, 0.05]

ACT_NAMES = {0: "TRANSIENT", 1: "SITTING", 2: "STAIRS", 3: "TABLE_SOCCER",
             4: "CYCLING", 5: "DRIVING", 6: "LUNCH", 7: "WALKING",
             8: "WORKING"}


def clean_rr(rr, tol, kernel=5):
    rr = np.asarray(rr, float)
    if len(rr) < kernel + 2:
        return np.empty(0), 1.0
    ok = (rr > 300) & (rr < 2000)
    if ok.sum() < kernel + 2:
        return np.empty(0), 1.0
    rp = rr[ok]
    med = medfilt(rp, kernel_size=kernel)
    keep = np.abs(rp - med) <= tol * med
    return rp[keep], float(1.0 - keep.sum() / len(rr))


def stats(rr):
    d = np.diff(rr)
    return (float(np.sqrt(np.mean(d ** 2))),
            60000.0 / float(np.mean(rr)),
            float(np.std(rr, ddof=1)))


def bvp_peak_times(bvp, fs=FS_BVP):
    b, a = butter(3, [0.7 / (fs / 2), 3.5 / (fs / 2)], btype="band")
    f = filtfilt(b, a, np.asarray(bvp, float).ravel())
    s = f.std()
    if s <= 0:
        return np.empty(0)
    f = f / s
    pk, _ = find_peaks(f, distance=int(0.35 * fs), prominence=0.5)
    pk = pk[(pk > 0) & (pk < len(f) - 1)]
    if len(pk) == 0:
        return np.empty(0)
    y0, y1, y2 = f[pk - 1], f[pk], f[pk + 1]
    den = y0 - 2 * y1 + y2
    sh = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
    return (pk + np.clip(sh, -0.5, 0.5)) / fs


def motion_series(acc, fs=FS_ACC):
    g = np.asarray(acc, float) / 64.0
    b, a = butter(2, 0.3 / (fs / 2), btype="high")
    return np.linalg.norm(filtfilt(b, a, g, axis=0), axis=1)


def run():
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    with zipfile.ZipFile(INNER) as z:
        targets = sorted((n for n in z.namelist()
                          if n.endswith(".pkl") and "/S" in n),
                         key=lambda n: int(Path(n).stem[1:]))
        for name in targets:
            sid = Path(name).stem
            dest = TMP / f"{sid}.pkl"
            with z.open(name) as src, open(dest, "wb") as fh:
                while chunk := src.read(1 << 22):
                    fh.write(chunk)
            with open(dest, "rb") as fh:
                d = pickle.load(fh, encoding="latin1")

            rpk = np.asarray(d["rpeaks"], float).ravel() / FS_ECG
            bvp = np.asarray(d["signal"]["wrist"]["BVP"], float).ravel()
            acc = np.asarray(d["signal"]["wrist"]["ACC"], float)
            act = np.asarray(d["activity"], float).ravel()
            ppk = bvp_peak_times(bvp)
            mot = motion_series(acc)
            d = None

            dur = min(rpk[-1] if len(rpk) else 0, len(bvp) / FS_BVP)
            for s0 in np.arange(0, dur - WIN, HOP):
                s1 = s0 + WIN
                er = np.diff(rpk[(rpk >= s0) & (rpk < s1)]) * 1000.0
                ec, erej = clean_rr(er, ECG_TOL)
                if len(ec) < MIN_BEATS or erej > MAX_REJECT:
                    continue
                e_rmssd, e_hr, _ = stats(ec)

                pr = np.diff(ppk[(ppk >= s0) & (ppk < s1)]) * 1000.0
                a = act[int(s0 * 4):int(s1 * 4)]
                m = mot[int(s0 * FS_ACC):int(s1 * FS_ACC)]
                base = dict(
                    subject=sid, t_start=float(s0),
                    activity=ACT_NAMES.get(
                        int(pd.Series(a).mode()[0]) if len(a) else -1, "NA"),
                    motion_sd=float(np.std(m)) if len(m) else np.nan,
                    ecg_rmssd=e_rmssd, ecg_hr=e_hr)

                for tol in TOLS:
                    pc, prej = clean_rr(pr, tol)
                    if len(pc) < MIN_BEATS or prej > MAX_REJECT:
                        rows.append({**base, "tol": tol, "kept": 0,
                                     "ppg_rmssd": np.nan, "ppg_hr": np.nan,
                                     "ppg_reject": prej})
                    else:
                        p_rmssd, p_hr, _ = stats(pc)
                        rows.append({**base, "tol": tol, "kept": 1,
                                     "ppg_rmssd": p_rmssd, "ppg_hr": p_hr,
                                     "ppg_reject": prej})

            n_win = len({r["t_start"] for r in rows if r["subject"] == sid})
            print(f"  {sid}: {dur/60:6.1f} min  {n_win:4d} ECG-valid windows",
                  flush=True)
            dest.unlink(missing_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ppg_sweep.csv", index=False)
    print(f"\nwrote {OUT/'ppg_sweep.csv'}  ({len(df):,} rows)")
    return df


def report(df):
    total = df[df.tol == TOLS[0]].shape[0]
    print(f"\nECG-valid windows: {total:,}\n")
    print(f"{'tol':>7}{'retained':>10}{'coverage':>10}{'bias':>9}"
          f"{'MAE':>8}{'r':>8}{'HR MAE':>9}")
    print("-" * 61)
    for tol in TOLS:
        s = df[(df.tol == tol) & (df.kept == 1)]
        if len(s) < 30:
            print(f"{tol:>7.3f}{len(s):>10}   too few")
            continue
        d = s.ppg_rmssd - s.ecg_rmssd
        print(f"{tol:>7.3f}{len(s):>10,}{100*len(s)/total:>9.1f}%"
              f"{d.mean():>+9.1f}{d.abs().mean():>8.1f}"
              f"{s.ecg_rmssd.corr(s.ppg_rmssd):>8.3f}"
              f"{(s.ppg_hr - s.ecg_hr).abs().mean():>9.2f}")
    print("\nbias/MAE in ms, HR MAE in bpm; coverage is % of ECG-valid windows")


if __name__ == "__main__":
    report(run())

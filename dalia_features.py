"""
PPG-DaLiA — window-level feature table for Paper 1.

Question: can wrist EDA be estimated from cardiac + motion + temperature
signals available on the same wrist device?

SOURCE OF TRUTH
---------------
Everything is read from the per-subject .pkl, because inside it the chest and
wrist streams are already time-aligned by the dataset authors. Two HRV sources
are computed so the paper can report both:

  ECG-HRV   from `rpeaks` -- chest ECG R-peaks at 700 Hz. Gold standard.
  PPG-HRV   from wrist BVP peaks. What a wrist product could actually compute.

Why this matters: a naive find_peaks on wrist BVP gives RMSSD ~270 ms, which
is physiologically impossible (daily-life range is 20-60 ms). It counts motion
ripple as beats. Even after artifact correction the PPG estimate carries a
~+30 ms RMSSD bias relative to ECG. Reporting both, and the gap between them,
is a result in its own right for a wrist-device paper.

Sampling rates (per dataset documentation):
  chest ECG 700 Hz | wrist BVP 64 Hz, ACC 32 Hz, EDA 4 Hz, TEMP 4 Hz
  activity labels 4 Hz

The .pkl files are ~1.4 GB each. Each is extracted, reduced to features, then
deleted, so peak disk use stays ~1.5 GB rather than 22 GB.

Writes ./dalia_out/windows.csv
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

FS_ECG, FS_BVP, FS_ACC, FS_EDA, FS_TMP, FS_ACT = 700, 64, 32, 4, 4, 4
WIN, HOP = 60.0, 10.0
MIN_BEATS, MAX_REJECT = 25, 0.20

ACT_NAMES = {0: "TRANSIENT", 1: "SITTING", 2: "STAIRS", 3: "TABLE_SOCCER",
             4: "CYCLING", 5: "DRIVING", 6: "LUNCH", 7: "WALKING",
             8: "WORKING"}


# ══════════════════════════════════════════════════════════════════
def clean_rr(rr, tol=0.25, kernel=5):
    """Reject intervals deviating >tol from the local median.

    Catches both extra peaks (RR halves) and missed beats (RR doubles);
    either one inflates RMSSD badly because RMSSD is built from successive
    differences.
    """
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


def hrv(rr_raw):
    rr, rej = clean_rr(rr_raw)
    if len(rr) < MIN_BEATS or rej > MAX_REJECT:
        return None
    d = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(d ** 2)))
    sd1 = rmssd / np.sqrt(2)
    var = float(np.var(rr, ddof=1))
    return dict(hr=60000.0 / float(np.mean(rr)),
                sdnn=float(np.std(rr, ddof=1)),
                rmssd=rmssd,
                pnn50=100.0 * float(np.mean(np.abs(d) > 50)),
                sd1=sd1,
                sd2=float(np.sqrt(max(2 * var - sd1 ** 2, 0.0))),
                n_beats=len(rr), reject=rej)


def bvp_peak_times(bvp, fs=FS_BVP):
    """Cardiac-band filter, then peak picking with parabolic refinement.

    Sub-sample refinement matters: at 64 Hz one sample is 15.6 ms, and RMSSD
    in daily life is only 20-60 ms, so quantisation alone is a large error.
    """
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
    denom = (y0 - 2 * y1 + y2)
    shift = np.where(np.abs(denom) > 1e-9, 0.5 * (y0 - y2) / denom, 0.0)
    return (pk + np.clip(shift, -0.5, 0.5)) / fs


def split_eda(eda, fs=FS_EDA, cutoff=0.05):
    if len(eda) < 30:
        return eda, np.zeros_like(eda)
    b, a = butter(2, cutoff / (fs / 2), btype="low")
    tonic = filtfilt(b, a, eda)
    return tonic, eda - tonic


def to_g(acc):
    """Convert wrist acceleration to g, inferring the scale from gravity.

    Empatica exports appear at two scales depending on the distribution:
    PPG-DaLiA ships values already in g (range about -2..2), while WESAD
    ships raw counts (-128..127, i.e. 64 counts per g). Hard-coding a
    divisor silently rescales one of them -- an earlier version of this
    file divided the DaLiA data by 64 and reported dynamic acceleration
    64x too small. The ordering across activities was unaffected, which is
    why the error survived inspection.

    Gravity settles it: a wrist accelerometer must read about 1 g in
    magnitude overall, so the divisor is whichever makes that true.
    """
    a = np.asarray(acc, float)
    med = float(np.median(np.linalg.norm(a, axis=1)))
    div = 1.0 if med < 4.0 else 64.0
    scaled = a / div
    check = float(np.median(np.linalg.norm(scaled, axis=1)))
    if not 0.7 <= check <= 1.3:
        raise ValueError(
            f"acceleration scale not recognised: raw median |a| = {med:.2f}, "
            f"after /{div:g} it is {check:.2f} g, expected about 1 g")
    return scaled


def motion(acc, fs=FS_ACC):
    """Strip gravity, return dynamic acceleration magnitude in g."""
    g = to_g(acc)
    b, a = butter(2, 0.3 / (fs / 2), btype="high")
    return np.linalg.norm(filtfilt(b, a, g, axis=0), axis=1)


# ══════════════════════════════════════════════════════════════════
def process(sid: str, d: dict) -> pd.DataFrame:
    w = d["signal"]["wrist"]
    eda = np.asarray(w["EDA"], float).ravel()
    temp = np.asarray(w["TEMP"], float).ravel()
    bvp = np.asarray(w["BVP"], float).ravel()
    acc = np.asarray(w["ACC"], float)
    act = np.asarray(d["activity"], float).ravel()
    rp = np.asarray(d["rpeaks"], float).ravel()
    q = d.get("questionnaire", {})

    ecg_t = rp / FS_ECG
    ecg_rr = np.diff(ecg_t) * 1000.0
    ecg_mid = ecg_t[:-1]

    ppg_t = bvp_peak_times(bvp)
    ppg_rr = np.diff(ppg_t) * 1000.0
    ppg_mid = ppg_t[:-1]

    mot = motion(acc)
    tonic, phasic = split_eda(eda)

    dur = min(len(eda) / FS_EDA, len(acc) / FS_ACC, len(act) / FS_ACT)
    rows = []
    for w0 in np.arange(0, dur - WIN, HOP):
        w1 = w0 + WIN
        se, ee = int(w0 * FS_EDA), int(w1 * FS_EDA)
        sa, ea = int(w0 * FS_ACC), int(w1 * FS_ACC)
        sc, ec = int(w0 * FS_ACT), int(w1 * FS_ACT)
        if ee > len(eda) or ea > len(acc) or ec > len(act):
            break

        codes = act[sc:ec].astype(int)
        code = int(np.bincount(codes[codes >= 0]).argmax()) if len(codes) else -1

        row = dict(subject=sid, t_start=float(w0),
                   activity=ACT_NAMES.get(code, f"CODE{code}"),
                   eda_mean=float(eda[se:ee].mean()),
                   eda_tonic=float(tonic[se:ee].mean()),
                   eda_phasic_sd=float(phasic[se:ee].std()),
                   temp=float(temp[se:ee].mean()),
                   motion=float(mot[sa:ea].mean()),
                   motion_sd=float(mot[sa:ea].std()),
                   age=float(q.get("AGE", np.nan)),
                   gender=str(q.get("Gender", "?")),
                   skin=str(q.get("SKIN", "?")),
                   sport=str(q.get("SPORT", "?")))

        h = hrv(ecg_rr[(ecg_mid >= w0) & (ecg_mid < w1)])
        if h:
            row.update({f"ecg_{k}": v for k, v in h.items()})
        h2 = hrv(ppg_rr[(ppg_mid >= w0) & (ppg_mid < w1)])
        if h2:
            row.update({f"ppg_{k}": v for k, v in h2.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    ec = df.get("ecg_hr", pd.Series(dtype=float))
    pc = df.get("ppg_hr", pd.Series(dtype=float))
    print(f"  {sid:<4} {dur/60:6.1f} min {len(df):>5} win   "
          f"ECG {100*ec.notna().mean():5.1f}%  PPG {100*pc.notna().mean():5.1f}%   "
          f"RMSSD ecg {df.get('ecg_rmssd', pd.Series([np.nan])).mean():5.1f} "
          f"ppg {df.get('ppg_rmssd', pd.Series([np.nan])).mean():5.1f} ms   "
          f"EDA {df.eda_mean.mean():5.2f} uS")
    return df


def main():
    OUT.mkdir(exist_ok=True)
    TMP.mkdir(exist_ok=True)
    print(f"window {WIN:g}s hop {HOP:g}s   artifact tol 25%   "
          f"max reject {MAX_REJECT:.0%}\n")

    frames = []
    with zipfile.ZipFile(INNER) as z:
        targets = sorted(
            (n for n in z.namelist()
             if n.endswith(".pkl") and "/S" in n),
            key=lambda n: int(Path(n).stem[1:]))
        for name in targets:
            sid = Path(name).stem
            tmp = TMP / f"{sid}.pkl"
            with z.open(name) as src, open(tmp, "wb") as out:
                while chunk := src.read(1 << 22):
                    out.write(chunk)
            with open(tmp, "rb") as f:
                d = pickle.load(f, encoding="latin1")
            frames.append(process(sid, d))
            del d
            tmp.unlink()

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "windows.csv", index=False)
    print(f"\nwrote {OUT/'windows.csv'}   {len(df):,} windows, "
          f"{df.subject.nunique()} subjects")

    print("\n--- HRV sanity (daily life: SDNN 30-100, RMSSD 20-60, pNN50 5-30) ---")
    print(f"{'metric':<14}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}{'coverage':>11}")
    for c in ["ecg_hr", "ecg_sdnn", "ecg_rmssd", "ecg_pnn50",
              "ppg_hr", "ppg_sdnn", "ppg_rmssd", "ppg_pnn50"]:
        if c in df and df[c].notna().any():
            print(f"{c:<14}{df[c].mean():>9.1f}{df[c].std():>9.1f}"
                  f"{df[c].min():>9.1f}{df[c].max():>9.1f}"
                  f"{100*df[c].notna().mean():>10.1f}%")

    both = df.dropna(subset=["ecg_hr", "ppg_hr"])
    if len(both) > 100:
        print(f"\n--- PPG-HRV vs ECG-HRV ({len(both):,} overlapping windows) ---")
        print(f"{'metric':<9}{'r':>9}{'bias':>10}{'MAE':>9}")
        for m in ("hr", "sdnn", "rmssd", "pnn50"):
            r = both[f"ecg_{m}"].corr(both[f"ppg_{m}"])
            bias = (both[f"ppg_{m}"] - both[f"ecg_{m}"]).mean()
            mae = (both[f"ppg_{m}"] - both[f"ecg_{m}"]).abs().mean()
            print(f"{m:<9}{r:>+9.3f}{bias:>+10.2f}{mae:>9.2f}")

    print("\n--- activity distribution ---")
    print(df.activity.value_counts().to_string())
    print("\n--- motion by activity (gravity removed, g) ---")
    print(df.groupby("activity").motion.mean().sort_values(
        ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()

"""
WESAD window-level features, with the R-peak detection this dataset forces
on us and the validation gates that decide whether the result is usable.

WESAD supplies raw 700 Hz chest ECG but no R-peak indices, unlike PPG-DaLiA.
Detection is therefore ours, and it is the step most likely to produce a
plausible-looking wrong answer: two earlier errors in this project (RMSSD of
270 ms from naive peak-picking, and a +61 ms bias that turned out to be a
rejection-threshold artifact) both came from peak handling. So detection uses
a Pan-Tompkins-style pipeline rather than find_peaks on the raw trace, and
nothing downstream is reported until three gates pass:

  GATE 1  cohort median RMSSD falls in the physiological 20-60 ms band
  GATE 2  heart rate is higher under stress than baseline -- the TSST is a
          validated stressor, so this is an external check on our detector
          rather than a finding
  GATE 3  electrodermal level is higher under stress than baseline

Gates 2 and 3 are the useful ones: they can only be passed by a pipeline that
is actually measuring what it claims to measure.

Windows never straddle a condition boundary. Labels: 1 baseline, 2 stress,
3 amusement, 4 meditation; everything else is discarded per the dataset
documentation.

Writes ./wesad_out/windows.csv
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, medfilt

DS = Path(__file__).parent.parent / "datasets"
ZIP = DS / "WESAD.zip"
TMP = DS / "wesad_tmp"
OUT = Path(__file__).parent / "wesad_out"

FS_CHEST, FS_BVP, FS_ACC, FS_EDA = 700, 64, 32, 4
WIN, HOP = 60.0, 10.0
MIN_BEATS, MAX_REJECT = 25, 0.20
COND = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def detect_rpeaks(ecg, fs=FS_CHEST):
    """Pan-Tompkins style: band-pass, derivative, square, integrate, threshold.

    Returns peak times in seconds, refined to the local maximum of the
    band-passed signal so that the timing is the R apex rather than the
    centre of the integration window -- RR intervals are the whole point
    here and the integrator shifts them systematically.
    """
    x = np.asarray(ecg, float).ravel()
    b, a = butter(3, [5.0 / (fs / 2), 15.0 / (fs / 2)], btype="band")
    f = filtfilt(b, a, x)
    d = np.diff(f, prepend=f[0])
    sq = d ** 2
    win = int(0.150 * fs)
    integ = np.convolve(sq, np.ones(win) / win, mode="same")
    thr = np.percentile(integ, 98) * 0.35
    cand, _ = find_peaks(integ, height=thr, distance=int(0.25 * fs))
    if len(cand) == 0:
        return np.empty(0)
    half = int(0.05 * fs)
    peaks = []
    for c in cand:
        lo, hi = max(c - half, 0), min(c + half, len(f))
        if hi > lo:
            peaks.append(lo + int(np.argmax(np.abs(f[lo:hi]))))
    peaks = np.unique(peaks)
    return peaks / fs


def clean_rr(rr, tol=0.25, kernel=5):
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
    var = float(np.var(rr, ddof=1))
    sd1 = rmssd / np.sqrt(2)
    return dict(hr=60000.0 / float(np.mean(rr)), sdnn=float(np.std(rr, ddof=1)),
                rmssd=rmssd, pnn50=100.0 * float(np.mean(np.abs(d) > 50)),
                sd1=sd1, sd2=float(np.sqrt(max(2 * var - sd1 ** 2, 0.0))),
                n_beats=len(rr), reject=rej)


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


def to_g(acc):
    """Convert wrist acceleration to g, inferring the scale from gravity.

    WESAD ships raw Empatica counts (-128..127, 64 per g); PPG-DaLiA ships
    values already in g. The same helper is used for both datasets so that
    a motion figure from one is comparable with the other -- assuming a
    single fixed divisor is what made the two incomparable in the first
    place.
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


def motion_series(acc, fs=FS_ACC):
    g = to_g(acc)
    b, a = butter(2, 0.3 / (fs / 2), btype="high")
    return np.linalg.norm(filtfilt(b, a, g, axis=0), axis=1)


def run():
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(ZIP) as z:
        pkls = sorted((n for n in z.namelist() if n.endswith(".pkl")),
                      key=lambda n: int(Path(n).stem[1:]))
        for name in pkls:
            sid = Path(name).stem
            dest = TMP / f"{sid}.pkl"
            with z.open(name) as src, open(dest, "wb") as fh:
                while chunk := src.read(1 << 22):
                    fh.write(chunk)
            with open(dest, "rb") as fh:
                d = pickle.load(fh, encoding="latin1")

            lab = np.asarray(d["label"]).ravel()
            ecg = np.asarray(d["signal"]["chest"]["ECG"]).ravel()
            c_eda = np.asarray(d["signal"]["chest"]["EDA"]).ravel()
            w_eda = np.asarray(d["signal"]["wrist"]["EDA"]).ravel()
            w_tmp = np.asarray(d["signal"]["wrist"]["TEMP"]).ravel()
            bvp = np.asarray(d["signal"]["wrist"]["BVP"]).ravel()
            acc = np.asarray(d["signal"]["wrist"]["ACC"], float)
            d = None

            rpk = detect_rpeaks(ecg)
            ppk = bvp_peak_times(bvp)
            mot = motion_series(acc)
            dur = len(lab) / FS_CHEST

            n_sub = 0
            for s0 in np.arange(0, dur - WIN, HOP):
                s1 = s0 + WIN
                seg = lab[int(s0 * FS_CHEST):int(s1 * FS_CHEST)]
                u = np.unique(seg)
                if len(u) != 1 or int(u[0]) not in COND:
                    continue                      # never straddle conditions
                cond = COND[int(u[0])]

                e = hrv(np.diff(rpk[(rpk >= s0) & (rpk < s1)]) * 1000.0)
                if e is None:
                    continue
                p = hrv(np.diff(ppk[(ppk >= s0) & (ppk < s1)]) * 1000.0)

                we = w_eda[int(s0 * FS_EDA):int(s1 * FS_EDA)]
                ce = c_eda[int(s0 * FS_CHEST):int(s1 * FS_CHEST)]
                wt = w_tmp[int(s0 * FS_EDA):int(s1 * FS_EDA)]
                mm = mot[int(s0 * FS_ACC):int(s1 * FS_ACC)]
                if len(we) < 60 or len(ce) < 1000:
                    continue

                rows.append(dict(
                    subject=sid, t_start=float(s0), condition=cond,
                    wrist_eda=float(np.mean(we)),
                    chest_eda=float(np.mean(ce)),
                    temp=float(np.mean(wt)) if len(wt) else np.nan,
                    motion=float(np.mean(mm)) if len(mm) else np.nan,
                    motion_sd=float(np.std(mm)) if len(mm) else np.nan,
                    ecg_hr=e["hr"], ecg_sdnn=e["sdnn"], ecg_rmssd=e["rmssd"],
                    ecg_pnn50=e["pnn50"], ecg_reject=e["reject"],
                    ppg_hr=p["hr"] if p else np.nan,
                    ppg_rmssd=p["rmssd"] if p else np.nan))
                n_sub += 1

            print(f"  {sid}: {dur/60:5.1f} min, {len(rpk):6d} R-peaks, "
                  f"{n_sub:4d} windows", flush=True)
            dest.unlink(missing_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "windows.csv", index=False)
    print(f"\nwrote {OUT/'windows.csv'}  ({len(df):,} windows, "
          f"{df.subject.nunique()} subjects)")
    return df


def gates(df):
    print("\n" + "=" * 66)
    print("VALIDATION GATES -- nothing downstream is reported unless all pass")
    print("=" * 66)

    med = df.ecg_rmssd.median()
    g1 = 20.0 <= med <= 60.0
    print(f"\nGATE 1  cohort median RMSSD = {med:.1f} ms "
          f"(physiological 20-60)  ->  {'PASS' if g1 else 'FAIL'}")
    print(f"        mean {df.ecg_rmssd.mean():.1f}, "
          f"IQR {df.ecg_rmssd.quantile(.25):.1f}-{df.ecg_rmssd.quantile(.75):.1f}, "
          f"mean HR {df.ecg_hr.mean():.1f} bpm, "
          f"median rejected {100*df.ecg_reject.median():.1f}%")

    def paired(col):
        piv = (df[df.condition.isin(["baseline", "stress"])]
               .groupby(["subject", "condition"])[col].mean().unstack())
        piv = piv.dropna()
        return piv, (piv["stress"] - piv["baseline"])

    piv_hr, d_hr = paired("ecg_hr")
    g2 = d_hr.mean() > 0 and (d_hr > 0).sum() >= 0.8 * len(d_hr)
    print(f"\nGATE 2  HR stress - baseline = {d_hr.mean():+.1f} bpm, "
          f"higher in {int((d_hr > 0).sum())}/{len(d_hr)} subjects"
          f"  ->  {'PASS' if g2 else 'FAIL'}")

    piv_e, d_e = paired("wrist_eda")
    g3 = d_e.mean() > 0 and (d_e > 0).sum() >= 0.7 * len(d_e)
    print(f"GATE 3  wrist EDA stress - baseline = {d_e.mean():+.2f} uS, "
          f"higher in {int((d_e > 0).sum())}/{len(d_e)} subjects"
          f"  ->  {'PASS' if g3 else 'FAIL'}")
    _, d_ce = paired("chest_eda")
    print(f"        (chest EDA {d_ce.mean():+.2f} uS, "
          f"higher in {int((d_ce > 0).sum())}/{len(d_ce)})")

    print(f"\n{'ALL GATES PASS' if (g1 and g2 and g3) else 'GATES FAILED - STOP'}")

    print("\nper-condition means")
    print(df.groupby("condition")[
        ["ecg_hr", "ecg_rmssd", "wrist_eda", "chest_eda", "motion_sd"]]
        .agg(["mean", "count"]).round(2).to_string())
    return g1 and g2 and g3


if __name__ == "__main__":
    gates(run())

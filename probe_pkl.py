"""Extract one subject's .pkl and check whether ECG R-peaks are usable."""
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np

DS = Path(__file__).parent.parent / "datasets"
INNER = DS / "outer" / "data.zip"
TMP = DS / "pkl_probe"
TMP.mkdir(exist_ok=True)

target = "PPG_FieldStudy/S1/S1.pkl"
dest = TMP / "S1.pkl"

if not dest.exists():
    t = time.time()
    with zipfile.ZipFile(INNER) as z:
        info = z.getinfo(target)
        print(f"extracting {target}  ({info.file_size/1e6:.0f} MB)...")
        with z.open(target) as src, open(dest, "wb") as out:
            while chunk := src.read(1 << 22):
                out.write(chunk)
    print(f"  extracted in {time.time()-t:.0f} s")

t = time.time()
with open(dest, "rb") as f:
    d = pickle.load(f, encoding="latin1")
print(f"loaded in {time.time()-t:.0f} s")

print("\ntop-level keys:", list(d.keys()))
for k, v in d.items():
    if isinstance(v, dict):
        print(f"  {k}: dict -> {list(v.keys())}")
        for k2, v2 in v.items():
            if isinstance(v2, dict):
                print(f"     {k2}: {list(v2.keys())}")
    elif isinstance(v, np.ndarray):
        print(f"  {k}: ndarray {v.shape} {v.dtype}")
    else:
        print(f"  {k}: {type(v).__name__}")

if "rpeaks" in d:
    rp = np.asarray(d["rpeaks"]).ravel()
    print(f"\nrpeaks: {len(rp):,} entries, dtype {rp.dtype}")
    print(f"  first 8: {rp[:8]}")
    for fs in (700.0,):
        rr = np.diff(rp) / fs * 1000.0
        ok = rr[(rr > 300) & (rr < 2000)]
        if len(ok) < 100:
            continue
        dd = np.diff(ok)
        print(f"\n  assuming {fs:g} Hz sample indices:")
        print(f"    n RR      {len(ok):,}  ({100*len(ok)/max(len(rr),1):.1f}% plausible)")
        print(f"    mean RR   {ok.mean():.1f} ms  -> HR {60000/ok.mean():.1f} bpm")
        print(f"    SDNN      {ok.std(ddof=1):.1f} ms")
        print(f"    RMSSD     {np.sqrt(np.mean(dd**2)):.1f} ms")
        print(f"    pNN50     {100*np.mean(np.abs(dd)>50):.1f} %")
        print("\n    expected for daily life: SDNN 40-90, RMSSD 20-60, pNN50 5-30")

sz = dest.stat().st_size / 1e6
print(f"\none pkl on disk: {sz:.0f} MB  ->  15 subjects ~= {sz*15/1000:.1f} GB")

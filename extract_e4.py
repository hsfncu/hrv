"""Selectively extract the Empatica E4 wrist data from PPG-DaLiA.

The per-subject .pkl files are ~1.3-1.6 GB each (~20 GB total) because they
carry the raw 700 Hz chest recordings. The S*_E4.zip files are 2-3 MB each and
contain exactly what a wrist-worn device produces:

    BVP.csv   64 Hz   blood volume pulse   -> pulse peaks -> IBI -> HRV
    EDA.csv    4 Hz   electrodermal activity   <- our prediction target
    TEMP.csv   4 Hz   skin temperature
    ACC.csv   32 Hz   3-axis acceleration
    HR.csv     1 Hz   device-computed heart rate (already averaged)
    IBI.csv    --     device-detected inter-beat intervals

Each E4 csv starts with two header rows: initial UNIX timestamp, then sample
rate, then the samples.
"""

import zipfile
from pathlib import Path

DS = Path(__file__).parent.parent / "datasets"
INNER = DS / "outer" / "data.zip"
OUT = DS / "e4"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = (".csv",)

    with zipfile.ZipFile(INNER) as z:
        names = z.namelist()
        e4_zips = sorted(n for n in names if n.endswith("_E4.zip"))
        meta = sorted(n for n in names
                      if n.endswith(("_activity.csv", "_quest.csv")))
        print(f"found {len(e4_zips)} E4 archives, {len(meta)} metadata files")

        for n in meta:
            sub = Path(n).parent.name
            dest = OUT / sub
            dest.mkdir(exist_ok=True)
            (dest / Path(n).name).write_bytes(z.read(n))

        for n in e4_zips:
            sub = Path(n).parent.name
            dest = OUT / sub
            dest.mkdir(exist_ok=True)
            raw = z.read(n)
            tmp = dest / "_e4.zip"
            tmp.write_bytes(raw)
            with zipfile.ZipFile(tmp) as inner:
                members = [m for m in inner.namelist() if m.endswith(wanted)]
                for m in members:
                    (dest / Path(m).name).write_bytes(inner.read(m))
            tmp.unlink()
            sizes = {p.name: p.stat().st_size for p in dest.glob("*.csv")}
            print(f"  {sub:<5} {len(sizes)} files  "
                  + "  ".join(f"{k.replace('.csv',''):<5}{v/1024:>7.0f}KB"
                              for k, v in sorted(sizes.items())))

    total = sum(p.stat().st_size for p in OUT.rglob("*.csv"))
    print(f"\nextracted {total/1e6:.1f} MB to {OUT}")


if __name__ == "__main__":
    main()

"""
Figure 5 -- the ceiling, and the importance reversal between datasets.

Panel (a) is the paper's most consequential single number made visible: two
electrodes recording skin conductance from the same participant at the same
time, per-subject correlation, sorted. The spread -- and the one participant
below zero -- says more about why cross-channel estimation is hard than any
model comparison does.

Panel (b) puts the two permutation-importance rankings side by side. Motion
leads in free-living data and heart rate leads once movement is controlled,
which is the ordering an artifactual account of the free-living association
predicts in advance.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

CM = 1 / 2.54
W2 = 18.0 * CM
BLUE, ORANGE, DARK, GREY = "#1F6FA8", "#D2691E", "#22303A", "#9AA5AD"
LGREY = "#EDF0F2"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

w = pd.read_csv(HERE / "wesad_out" / "windows.csv")
w = w.dropna(subset=["wrist_eda", "chest_eda"])

# ── (a) per-subject wrist-chest EDA agreement ──
r = (w.groupby("subject")
     .apply(lambda g: g.wrist_eda.corr(g.chest_eda), include_groups=False)
     .sort_values())
pooled = w.wrist_eda.corr(w.chest_eda)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 7.0 * CM))

cols = [ORANGE if v < 0 else BLUE for v in r.values]
ax1.barh(np.arange(len(r)), r.values, color=cols, height=0.72)
ax1.set_yticks(np.arange(len(r)))
ax1.set_yticklabels(r.index, fontsize=6.5)
ax1.axvline(0, color=DARK, lw=0.7)
ax1.axvline(pooled, color=DARK, lw=0.9, ls="--")
ax1.annotate(f"pooled $r$ = {pooled:.3f}", xy=(pooled, len(r) - 0.4),
             xytext=(4, 0), textcoords="offset points",
             fontsize=6.5, color=DARK, va="center")
ax1.set_xlabel("Correlation between wrist and chest EDA ($r$)")
ax1.set_xlim(-0.8, 1.0)
ax1.set_ylim(-0.8, len(r) - 0.2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_title("(a) Two direct EDA measurements, same participant",
              loc="left")
ax1.annotate("opposite direction", xy=(r.values[0], 0),
             xytext=(6, -11), textcoords="offset points",
             fontsize=6.2, color=ORANGE)

# ── (b) permutation importance, both datasets ──
feats = ["Motion (SD)", "Heart rate", "Skin temp.", "SDNN",
         "Motion (mean)", "RMSSD", "pNN50"]
dalia = [0.1740, 0.0258, 0.0313, 0.0490, -0.0252, 0.0034, -0.0014]
wesad = [0.0517, 0.1665, 0.1440, 0.0042, 0.0227, -0.0397, -0.0254]

y = np.arange(len(feats))
h = 0.38
ax2.barh(y + h / 2, dalia, height=h, color=BLUE,
         label="PPG-DaLiA (free-living)")
ax2.barh(y - h / 2, wesad, height=h, color=ORANGE,
         label="WESAD (laboratory)")
ax2.set_yticks(y)
ax2.set_yticklabels(feats, fontsize=7)
ax2.invert_yaxis()
ax2.axvline(0, color=DARK, lw=0.7)
ax2.set_xlabel("Permutation importance (held-out)")
ax2.legend(frameon=False, loc="lower right")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_title("(b) What each model uses", loc="left")

fig.tight_layout()
for ext in ("pdf", "eps", "png", "tif"):
    kw = dict(dpi=600) if ext in ("png", "tif") else {}
    if ext == "tif":
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(FIG / f"fig5.{ext}", **kw)
plt.close(fig)

print(f"wrote fig5.*  pooled r = {pooled:.3f}, "
      f"per-subject mean {r.mean():.3f}, median {r.median():.3f}, "
      f"range [{r.min():.3f}, {r.max():.3f}], "
      f"positive {int((r > 0).sum())}/{len(r)}")

"""
Figure 4 -- accuracy/coverage trade-off for wrist-PPG RMSSD.

The point the figure has to make is that no threshold gives both. Panel (a)
plots bias and coverage against the rejection threshold on twin axes so the
scissors shape is unmissable; panel (b) plots agreement directly against
coverage, which is the trade-off a product decision actually faces, and shows
that agreement peaks at r = 0.585 and then falls again as windows run out.

Same house style as Figures 1-3: 8 pt type, colour-blind-safe blue/orange,
TrueType embedding, drawn at final print size.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).parent / "dalia_out"
FIG = Path(__file__).parent / "figures"
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

d = pd.read_csv(OUT / "ppg_sweep_summary.csv").sort_values("tol")

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(W2, 6.4 * CM))

# ── (a) bias vs coverage ──
# Both panels are drawn against coverage rather than against the threshold:
# coverage is the quantity a deployment decision is actually constrained by,
# and putting bias and coverage on twin axes made two monotone curves sit on
# top of each other, which reads as one series plotted twice.
ax1.axhspan(-10, 10, color=LGREY, zorder=0)
ax1.plot(d["cov"], d.bias, "o-", color=ORANGE, lw=1.3, ms=4)
ax1.axhline(0, color=GREY, lw=0.6, ls=":", zorder=1)
ax1.axhline(32.1, color=DARK, lw=0.6, ls="-.", zorder=1)
ax1.annotate("cohort mean ECG RMSSD = 32.1 ms (Table 1)", xy=(2, 34.5), fontsize=6.5,
             color=DARK, ha="left", va="bottom")
ax1.annotate("bias within $\\pm$10 ms", xy=(52, 11.5), fontsize=6.5,
             color="#6E7B84", ha="right", va="bottom")
for _, row in d.iterrows():
    ax1.annotate(f"{row.tol*100:.3g}%", (row["cov"], row.bias),
                 xytext=(4, -7), textcoords="offset points",
                 ha="left", fontsize=6, color=GREY)
ax1.set_xlabel("Window coverage (%)")
ax1.set_ylabel("RMSSD bias (ms)")
ax1.set_xlim(0, 55)
ax1.set_ylim(-10, 95)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_title("(a) Bias falls only as coverage collapses", loc="left")

# ── (b) agreement vs coverage ──
ax3.plot(d["cov"], d.r, "o-", color=DARK, lw=1.3, ms=4)
for _, row in d.iterrows():
    ax3.annotate(f"{row.tol*100:.3g}%", (row["cov"], row.r),
                 xytext=(0, -9), textcoords="offset points",
                 ha="center", fontsize=6, color=GREY)
ax3.set_xlabel("Window coverage (%)")
ax3.set_ylabel("Agreement with ECG RMSSD ($r$)")
ax3.set_ylim(0.25, 0.68)
ax3.set_xlim(0, 55)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.set_title("(b) Agreement never becomes usable", loc="left")
ax3.annotate("peak $r$ = 0.585\nat 11.0% coverage",
             xy=(11.0, 0.585), xytext=(17, 0.615),
             fontsize=6.5, color=DARK,
             arrowprops=dict(arrowstyle="->", lw=0.6, color=DARK))

fig.tight_layout()
for ext in ("pdf", "eps", "png", "tif"):
    kw = dict(dpi=600) if ext in ("png", "tif") else {}
    if ext == "tif":
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(FIG / f"fig4.{ext}", **kw)
fig.savefig(OUT / "fig4_coverage_tradeoff.png", dpi=600)
plt.close(fig)

print("wrote fig4.{pdf,eps,png,tif} and dalia_out/fig4_coverage_tradeoff.png")
print(d[["tol", "cov", "bias", "r"]].to_string(index=False))




"""
Publication-quality figures for Paper 1.

Outputs, for each figure, into ./figures/ :
    figN.pdf   vector, fonts embedded as TrueType  <- submit this
    figN.eps   vector, for journals that insist on EPS
    figN.png   600 dpi raster, for preview / Word
    figN.tif   600 dpi LZW-compressed, for journals that demand TIFF

DESIGN CONSTRAINTS APPLIED
--------------------------
* Sized to journal column widths, not to the screen. A figure drawn at
  14 inches and shrunk to 7 halves every font on the page; drawing it at
  final size is the only way the type ends up the size you chose.
      MDPI / Elsevier single column ~ 8.9 cm   double ~ 18 cm
* 8 pt base type, 7 pt ticks. At final print size this is the smallest
  that stays legible; most journals set 6 pt as the floor.
* pdf.fonttype / ps.fonttype = 42 embeds TrueType outlines. The default
  (Type 3) is rejected by IEEE and by several Elsevier titles.
* Colour-blind safe: the categorical pairs are blue/orange (distinguishable
  under deuteranopia and protanopia), never red/green. Sign is additionally
  encoded by bar direction, so colour is never the only channel.
* No transparency in the vector outputs -- alpha in EPS rasterises the
  whole panel at some publishers. The scatter panels use small opaque
  points with size chosen for density instead.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut

OUT = Path(__file__).parent / "dalia_out"
FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

CM = 1 / 2.54
W2 = 18.0 * CM          # double-column width
W1 = 8.9 * CM           # single-column width

# colour-blind safe pair + neutrals
BLUE = "#1F6FA8"
ORANGE = "#D2691E"
DARK = "#22303A"
GREY = "#9AA5AD"
LGREY = "#D5DBDF"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.4,
    "grid.color": LGREY,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,        # embed TrueType, not Type 3
    "ps.fonttype": 42,
    # Render maths in the same family as the body text. Matplotlib's default
    # mathtext font is DejaVu, which otherwise gets embedded alongside Arial
    # and makes "$R^2$" visibly different from the surrounding labels.
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name):
    for ext, kw in [("pdf", {}), ("eps", {}),
                    ("png", dict(dpi=600)),
                    ("tif", dict(dpi=600, pil_kwargs={"compression": "tiff_lzw"}))]:
        fig.savefig(FIG / f"{name}.{ext}", **kw)
    sizes = {e: (FIG / f"{name}.{e}").stat().st_size / 1024
             for e in ("pdf", "eps", "png", "tif")}
    print(f"  {name}: " + "  ".join(f"{k} {v:,.0f}KB" for k, v in sizes.items()))
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
df = pd.read_csv(OUT / "windows.csv")
CARD = ["ecg_hr", "ecg_sdnn", "ecg_rmssd", "ecg_pnn50"]
ALL = CARD + ["motion", "motion_sd", "temp"]
LABEL = {"ecg_hr": "HR", "ecg_sdnn": "SDNN", "ecg_rmssd": "RMSSD",
         "ecg_pnn50": "pNN50", "motion": "Motion mean",
         "motion_sd": "Motion SD", "temp": "Skin temp."}
df = df.dropna(subset=CARD + ["eda_mean", "temp", "motion"]).copy()
df["eda_within"] = df.groupby("subject").eda_mean.transform(
    lambda s: (s - s.mean()) / s.std())
g = df.groupby("subject").eda_mean.agg(["mean", "std"])

print("regenerating figures at publication quality\n")

# ══════════════════════════════════════════════════════════════════
# FIGURE 1  (three panels, double column)
# ══════════════════════════════════════════════════════════════════
X, y, grp = df[ALL].values, df.eda_within.values, df.subject.values
logo = LeaveOneGroupOut()
yt, yp, gs, acts = [], [], [], []
for tr, te in logo.split(X, y, grp):
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                              random_state=0, n_jobs=-1)
    m.fit(X[tr], y[tr])
    yt.append(y[te]); yp.append(m.predict(X[te]))
    gs.append(grp[te]); acts.append(df.activity.values[te])
yt, yp, gs, acts = map(np.concatenate, (yt, yp, gs, acts))
subs = np.unique(grp)
from sklearn.metrics import r2_score
per_sub = {s: r2_score(yt[gs == s], yp[gs == s]) for s in subs}

fig, ax = plt.subplots(1, 3, figsize=(W2, 6.0 * CM))

# (a) between-subject EDA
order = g["mean"].sort_values().index
ax[0].bar(range(len(order)), g.loc[order, "mean"], yerr=g.loc[order, "std"],
          color=BLUE, ecolor=GREY, capsize=1.5, width=0.72,
          error_kw=dict(elinewidth=0.5, capthick=0.5))
ax[0].set_xticks(range(len(order)))
ax[0].set_xticklabels(order, rotation=90, fontsize=5.5)
ax[0].set_xlabel("Participant", labelpad=2)
ax[0].set_ylabel("EDA (µS)")
ax[0].set_title("(a)  Between-participant SCL", loc="left", fontweight="bold")
ax[0].grid(axis="y", alpha=1)
ax[0].set_axisbelow(True)
ax[0].annotate(f"{g['mean'].max()/g['mean'].min():.1f}× range",
               xy=(0.04, 0.92), xycoords="axes fraction", fontsize=7,
               color=DARK)

# (b) pooled vs within r
pool = [df[c].corr(df.eda_mean) for c in ALL]
with_ = [df.groupby("subject").apply(
    lambda x, c=c: x[c].corr(x.eda_mean), include_groups=False).mean()
    for c in ALL]
xx = np.arange(len(ALL))
ax[1].bar(xx - .19, pool, .38, label="Pooled", color=ORANGE)
ax[1].bar(xx + .19, with_, .38, label="Within-participant", color=BLUE)
ax[1].axhline(0, color=DARK, lw=0.6)
ax[1].set_xticks(xx)
ax[1].set_xticklabels([LABEL[c] for c in ALL], fontsize=6.5,
                      rotation=38, ha="right", rotation_mode="anchor")
ax[1].set_ylabel("Correlation with EDA")
ax[1].set_title("(b)  Pooled vs within-participant", loc="left",
                fontweight="bold")
# headroom so the legend never sits on a bar
ax[1].set_ylim(min(min(pool), min(with_)) - 0.04,
               max(max(pool), max(with_)) + 0.13)
ax[1].legend(frameon=False, loc="upper right", ncol=2,
             handlelength=1.1, handletextpad=0.5, columnspacing=1.0,
             borderaxespad=0.1)
ax[1].grid(axis="y", alpha=1)
ax[1].set_axisbelow(True)

# (c) per-subject LOSO R2 -- sign encoded by direction AND colour
ps = sorted(per_sub.items(), key=lambda t: t[1])
cols = [ORANGE if v < 0 else BLUE for _, v in ps]
ax[2].barh([p[0] for p in ps], [p[1] for p in ps], color=cols, height=0.68)
ax[2].axvline(0, color=DARK, lw=0.6)
ax[2].set_xlabel("Leave-one-participant-out $R^2$")
ax[2].set_title("(c)  Per-participant generalisation", loc="left",
                fontweight="bold")
ax[2].tick_params(axis="y", labelsize=5.5)
ax[2].grid(axis="x", alpha=1)
ax[2].set_axisbelow(True)
ax[2].annotate(f"positive in {sum(v > 0 for v in per_sub.values())}/15",
               xy=(0.55, 0.06), xycoords="axes fraction", fontsize=7,
               color=DARK)

fig.tight_layout(w_pad=1.6)
save(fig, "figure1")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2  Bland-Altman (two panels, double column)
# ══════════════════════════════════════════════════════════════════
both = df.dropna(subset=["ppg_rmssd", "ecg_rmssd"])
fig, ax = plt.subplots(1, 2, figsize=(W2, 6.4 * CM))
for a, m, unit, tag in [(ax[0], "hr", "bpm", "(a)"),
                        (ax[1], "rmssd", "ms", "(b)")]:
    x = (both[f"ecg_{m}"] + both[f"ppg_{m}"]) / 2
    d = both[f"ppg_{m}"] - both[f"ecg_{m}"]
    bias, sd = d.mean(), d.std()
    a.scatter(x, d, s=1.2, color=BLUE, edgecolors="none", rasterized=True)
    a.axhline(bias, color=ORANGE, lw=1.1)
    a.axhline(bias + 1.96 * sd, color=ORANGE, ls=(0, (4, 2)), lw=0.8)
    a.axhline(bias - 1.96 * sd, color=ORANGE, ls=(0, (4, 2)), lw=0.8)
    a.axhline(0, color=DARK, lw=0.6)
    r = both[f"ecg_{m}"].corr(both[f"ppg_{m}"])
    a.set_xlabel(f"Mean of ECG and PPG ({unit})")
    a.set_ylabel(f"PPG − ECG ({unit})")
    a.set_title(f"{tag}  {m.upper()}", loc="left", fontweight="bold")
    a.grid(alpha=1)
    a.set_axisbelow(True)
    a.annotate(f"$r$ = {r:+.3f}\nbias = {bias:+.2f} {unit}\n"
               f"1.96 SD = {1.96*sd:.1f} {unit}",
               xy=(0.035, 0.96), xycoords="axes fraction", va="top",
               fontsize=6.5, color=DARK,
               bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=LGREY,
                         lw=0.5))
fig.tight_layout(w_pad=2.0)
save(fig, "figure2")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3  activity-stratified bias (two panels, double column)
# ══════════════════════════════════════════════════════════════════
rows = []
for a_ in df.activity.unique():
    mk = acts == a_
    if mk.sum() < 100:
        continue
    sub = df[df.activity == a_]
    rows.append(dict(activity=a_.replace("_", " ").title(), n=int(mk.sum()),
                     motion=sub.motion.mean(), hr=sub.ecg_hr.mean(),
                     true=yt[mk].mean(), pred=yp[mk].mean(),
                     bias=yp[mk].mean() - yt[mk].mean()))
r = pd.DataFrame(rows).sort_values("motion")
xx = np.arange(len(r))

fig, ax = plt.subplots(1, 2, figsize=(W2, 6.6 * CM))
ax[0].barh(xx - .19, r.true, .38, label="Measured", color=DARK)
ax[0].barh(xx + .19, r.pred, .38, label="Predicted", color=ORANGE)
ax[0].axvline(0, color=DARK, lw=0.6)
ax[0].set_yticks(xx)
ax[0].set_yticklabels(r.activity, fontsize=6.5)
ax[0].set_xlabel("EDA (within-participant SD)")
ax[0].set_title("(a)  Measured vs predicted", loc="left", fontweight="bold")
ax[0].legend(frameon=False, loc="lower right")
ax[0].grid(axis="x", alpha=1)
ax[0].set_axisbelow(True)

cols = [ORANGE if b > 0 else BLUE for b in r.bias]
ax[1].barh(xx, r.bias, color=cols, height=0.68)
ax[1].axvline(0, color=DARK, lw=0.6)
ax[1].set_yticks(xx)
ax[1].set_yticklabels([f"{a}  ({m*1000:.1f} mg)"
                       for a, m in zip(r.activity, r.motion)], fontsize=6)
ax[1].set_xlabel("Prediction bias (predicted − measured, SD)")
ax[1].set_title("(b)  Bias, ordered by wrist motion", loc="left",
                fontweight="bold")
ax[1].grid(axis="x", alpha=1)
ax[1].set_axisbelow(True)
ax[1].annotate("over-predicted →", xy=(0.62, 0.03), xycoords="axes fraction",
               fontsize=6, color=ORANGE)
ax[1].annotate("← under-predicted", xy=(0.04, 0.03), xycoords="axes fraction",
               fontsize=6, color=BLUE)
fig.tight_layout(w_pad=2.0)
save(fig, "figure3")

print(f"\nall figures written to {FIG}")
print("\nsubmit the .pdf files; .eps for journals that require it;")
print(".tif at 600 dpi where a raster is mandated; .png for Word drafts.")

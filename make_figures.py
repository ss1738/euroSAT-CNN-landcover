#!/usr/bin/env python3
"""Generate README figures from results/*.json (run from repo root)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY, RED = "#4269D0", "#EFB118", "#9AA0A6", "#C5443C"
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})

def L(p): return json.load(open(p))["summary"]

# 1. Band-selection curve (stage 7)
s7 = L("results/stage7_results.json")["band_curve"]
order = ["rgb3", "nir5", "b8", "all13"]
kb = [s7[k]["bytes_per_tile"] / 1024 for k in order]
acc = [s7[k]["acc"]["mean"] * 100 for k in order]
lbl = ["RGB (3)", "+NIR/SWIR (5)", "8 bands", "all 13"]
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(kb, acc, "-o", color=BLUE, lw=2, ms=8)
for x, y, t in zip(kb, acc, lbl):
    ax.annotate(f"{t}\n{y:.1f}%", (x, y), textcoords="offset points", xytext=(6, -18), fontsize=9)
ax.set_xlabel("sensor data per tile (KB)"); ax.set_ylabel("accuracy (%)")
ax.set_title("Accuracy per byte of sensor budget (n=3 seeds)")
fig.tight_layout(); fig.savefig("figures/band_curve.png"); plt.close(fig)

# 2. Coverage under quantization (stage 2, n=20)
s2 = L("results/sweep_results.json")["by_alpha"]
alphas = ["0.05", "0.1", "0.15"]
import numpy as np
x = np.arange(3); w = 0.26
fp = [s2[a]["cov_fp32"]["mean"] * 100 for a in alphas]
st = [s2[a]["cov_int8_stale"]["mean"] * 100 for a in alphas]
rc = [s2[a]["cov_int8_recal"]["mean"] * 100 for a in alphas]
tg = [95, 90, 85]
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.bar(x - w, fp, w, color=GRAY, label="FP32")
ax.bar(x, st, w, color=RED, label="INT8, stale calibration")
ax.bar(x + w, rc, w, color=BLUE, label="INT8, recalibrated")
for i, t in enumerate(tg):
    ax.hlines(t, i - 1.6 * w, i + 1.6 * w, color="black", lw=1.2, ls="--")
ax.set_xticks(x, [f"α={a}  (target {t}%)" for a, t in zip(alphas, tg)])
ax.set_ylabel("empirical coverage (%)"); ax.set_ylim(80, 97)
ax.set_title("Quantization breaks conformal coverage; recalibration restores it (n=20)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig("figures/coverage.png"); plt.close(fig)

# 3. Three-check gate residual risk (stage 5, n=20)
s5 = L("results/stage5_results.json")["gate20"]["teacher"]
conds = ["clean", "bright", "blur", "noise"]
risk = [s5[c]["err_accepted"]["mean"] * 100 for c in conds]
acp = [s5[c]["accept_rate"]["mean"] * 100 for c in conds]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
axes[0].bar(conds, acp, color=BLUE)
axes[0].set_ylabel("accepted (%)"); axes[0].set_title("Gate acceptance")
axes[1].bar(conds, risk, color=ORANGE)
axes[1].set_ylabel("error among accepted (%)"); axes[1].set_title("Residual risk")
axes[1].annotate("0 accepted", ("noise", 1.5), ha="center", fontsize=9)
fig.suptitle("Three-check gate under corruption (ResNet18 INT8, n=20)", y=1.02)
fig.tight_layout(); fig.savefig("figures/gate.png", bbox_inches="tight"); plt.close(fig)

# 4. Ladder + SEU (stage 6)
s6 = L("results/stage6_results.json")
lad = s6["ladder"]; seu = s6["seu_fp32_bitflips"]["teacher"]
sizes = [lad[wd]["int8_bytes"] / 1024 for wd in ["1.0", "0.5", "0.25"]]
lacc = [lad[wd]["clean"]["acc"]["mean"] * 100 for wd in ["1.0", "0.5", "0.25"]]
lrisk = [lad[wd]["clean"]["err_accepted"]["mean"] * 100 for wd in ["1.0", "0.5", "0.25"]]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
axes[0].plot(sizes, lacc, "-o", color=BLUE, label="clean accuracy")
axes[0].plot(sizes, lrisk, "-o", color=ORANGE, label="residual risk (accepted)")
axes[0].set_xlabel("INT8 model size (KB)"); axes[0].set_ylabel("%")
axes[0].set_title("Compression ladder (n=5)"); axes[0].legend(frameon=False, fontsize=9)
ks = ["0", "1", "10", "100"]
axes[1].plot(ks, [seu[k]["acc"]["mean"] * 100 for k in ks], "-o", color=BLUE, label="accuracy")
axes[1].plot(ks, [seu[k]["err_accepted"]["mean"] * 100 for k in ks], "-o", color=ORANGE,
             label="error among accepted")
axes[1].set_xlabel("weight bit flips"); axes[1].set_title("SEU injection, teacher (10 trials)")
axes[1].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig("figures/ladder_seu.png"); plt.close(fig)
print("figures written")

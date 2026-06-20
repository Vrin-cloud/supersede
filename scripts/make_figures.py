"""Generate paper figures (vector PDF) from the measured results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "research paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

models = ["gpt-4.1-mini", "gpt-4.1", "gpt-5.4"]
full = [82, 91, 92]
bounded = [63, 64, 77]

fig, ax = plt.subplots(figsize=(6.0, 3.6))
x = range(len(models))
w = 0.36
b1 = ax.bar([i - w / 2 for i in x], full, w, label="Full context",
           color="#9bbcd6", edgecolor="#33536e")
b2 = ax.bar([i + w / 2 for i in x], bounded, w, label="Bounded memory",
           color="#3a6ea5", edgecolor="#1f3b56")

for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
            f"{int(b.get_height())}", ha="center", fontsize=9)

# gap arrows to the right of the bounded bar, so they never touch the labels
for i in x:
    ax_x = i + w / 2 + w * 0.75
    ax.annotate("", xy=(ax_x, bounded[i]), xytext=(ax_x, full[i]),
                arrowprops=dict(arrowstyle="<->", color="#b03030", lw=1.1))
    ax.text(ax_x + 0.06, (full[i] + bounded[i]) / 2,
            f"$-{full[i] - bounded[i]}$", color="#b03030", fontsize=9, va="center")

ax.axhline(100, ls=":", lw=1, color="#888")
ax.text(2.4, 101, "synthetic in-context (100%)", fontsize=8,
        color="#888", ha="right")

ax.set_xticks(list(x))
ax.set_xticklabels(models)
ax.set_ylabel("Knowledge-update accuracy (%)")
ax.set_ylim(0, 112)
ax.set_xlim(-0.6, 2.7)
ax.legend(loc="lower left", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "results.pdf", bbox_inches="tight")
print("wrote", OUT / "results.pdf")

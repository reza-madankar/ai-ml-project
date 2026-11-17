"""Run: uv run --with matplotlib python scripts/plot_hybrid_tuning.py."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parents[1] / "reports/hybrid-tuning"
report = json.loads((root / "summary.json").read_text())
grid = report["development_grid"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
axes[0].plot([row["alpha"] for row in grid], [row["ndcg@10"] for row in grid], marker="o", color="#286ca3")
axes[0].axvline(report["selected_alpha"], color="#d86a25", linestyle="--",
                label=f"Selected alpha: {report['selected_alpha']:.2f}")
axes[0].set(title="Weight selection: development questions only",
            xlabel="BM25 weight alpha (embedding weight = 1 - alpha)", ylabel="Passage nDCG@10")
axes[0].legend()
strategies = ["bm25", "embedding", "hybrid"]
scores = [report["held_out"][name]["ndcg@10"] for name in strategies]
bars = axes[1].bar(["BM25", "Embedding", "Tuned hybrid"], scores,
                   color=["#7c8797", "#286ca3", "#29834d"])
axes[1].bar_label(bars, labels=[f"{score:.4f}" for score in scores], padding=4)
axes[1].set(title=f"Held-out evaluation ({report['coverage']['held_out']['questions']:,} questions)",
            ylabel="Passage nDCG@10", ylim=(0, 1))
for axis in axes:
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
fig.suptitle("BioASQ: BM25 + Perplexity embeddings · 500-word chunks / 75-word overlap")
fig.savefig(root / "weight_tuning.png", dpi=180)
fig.savefig(root / "weight_tuning.svg")

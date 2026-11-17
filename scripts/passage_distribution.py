"""Run with: uv run --with matplotlib python scripts/passage_distribution.py."""

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/rag-mini-bioasq/data/passages.parquet/part.0.parquet"
OUTPUT = ROOT / "reports/passage-distribution"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(SOURCE, columns=["id", "passage"])
    rows = []
    usable_rows = []
    skipped = 0
    for row in table.to_pylist():
        text = row["passage"]
        if not isinstance(text, str):
            skipped += 1
            continue
        rows.append((row["id"], len(text.split()), len(text), len(text.encode("utf-8"))))
        if text.strip() and text.strip().lower() != "nan":
            usable_rows.append(rows[-1])
    with (OUTPUT / "passage_lengths.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["passage_id", "words", "characters", "utf8_bytes"])
        writer.writerows(rows)

    values = np.array([row[1:] for row in rows])
    summary = {"passages": len(rows), "skipped_non_text": skipped,
               "definitions": {"words": "Whitespace-separated words, same as chunking",
                               "characters": "Unicode code points, including whitespace",
                               "utf8_bytes": "UTF-8 encoded text only; excludes file overhead"}}
    for index, name in enumerate(["words", "characters", "utf8_bytes"]):
        data = values[:, index]
        summary[name] = {"min": int(data.min()), "mean": float(data.mean()),
                         "std": float(data.std()),
                         **{f"p{p}": float(np.percentile(data, p))
                            for p in [25, 50, 75, 90, 95, 99]},
                         "max": int(data.max()), "total": int(data.sum())}
    edges = [0, 50, 100, 150, 200, 300, 500, 1000, float("inf")]
    distribution = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        count = int(((values[:, 0] >= lower) & (values[:, 0] < upper)).sum())
        label = f"{lower}–{int(upper) - 1}" if np.isfinite(upper) else f"{lower}+"
        distribution.append({"words": label, "count": count, "percent": 100 * count / len(rows)})
    summary["word_count_bins"] = distribution
    summary["passages_over_150_words"] = int((values[:, 0] > 150).sum())
    summary["passages_over_500_words"] = int((values[:, 0] > 500).sum())
    summary["parquet_file_bytes"] = SOURCE.stat().st_size
    summary["literal_nan_passages"] = len(rows) - len(usable_rows)
    usable = np.array([row[1:] for row in usable_rows])
    summary["excluding_literal_nan"] = {"passages": len(usable_rows)}
    for index, name in enumerate(["words", "characters", "utf8_bytes"]):
        data = usable[:, index]
        summary["excluding_literal_nan"][name] = {
            "min": int(data.min()), "mean": float(data.mean()),
            **{f"p{p}": float(np.percentile(data, p)) for p in [25, 50, 75, 90, 95, 99]},
            "max": int(data.max())}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for index, (axis, title) in enumerate(zip(axes, ["Words per passage", "Characters per passage", "UTF-8 bytes per passage"])):
        data = values[:, index]
        positive = data[data > 0]
        bins = np.geomspace(max(1, positive.min()), positive.max() + 1, 55)
        axis.hist(positive, bins=bins, color="#286ca3", edgecolor="white", linewidth=0.3)
        axis.set_xscale("log")
        axis.axvline(np.median(data), color="#d86a25", label=f"Median: {np.median(data):,.0f}")
        if index == 0:
            axis.axvline(500, color="#29834d", linestyle="--", label="Chunk size: 500 words")
        axis.set(title=title, xlabel="Length (log scale)", ylabel="Passages")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"BioASQ original passages · {len(rows):,} passages (includes {len(rows) - len(usable_rows):,} literal 'nan' entries)\nEqual-width logarithmic bins")
    fig.savefig(OUTPUT / "distribution.png", dpi=180)
    fig.savefig(OUTPUT / "distribution.svg")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

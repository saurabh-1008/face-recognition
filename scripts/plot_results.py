"""Plot measured size, latency and pair scores from a comparison report."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "results/sanity_comparison.json"))
    parser.add_argument("--output", default=str(ROOT / "results/graphs/model_comparison"))
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    models = report["models"]
    names = list(models)
    colors = ["#2563eb", "#0891b2", "#059669", "#d97706"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle("InsightFace model comparison", fontsize=19, fontweight="bold", y=0.98)
    for ax, key, title, unit in zip(axes[:2],
            ["onnx_size_mb", "mean_image_seconds"],
            ["Downloaded ONNX size", "Processing time per image"],
            ["Decimal MB", "Milliseconds"]):
        for i, name in enumerate(names):
            value = models[name].get(key)
            if value is None:
                ax.text(i, 0, "Unavailable", rotation=90, va="bottom", ha="center")
                continue
            value *= 1000 if key == "mean_image_seconds" else 1
            ax.bar(i, value, color=colors[i % len(colors)], width=0.65)
            ax.annotate(f"{value:.2f}" if key == "onnx_size_mb" else f"{value:.1f}",
                        (i, value), xytext=(0, 5), textcoords="offset points", ha="center")
        ax.set_title(title, pad=15)
        ax.set_ylabel(unit)
        ax.set_xticks(range(len(names)), names, rotation=20)
        ax.set_ylim(0, max(ax.get_ylim()[1] * 1.16, 1))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    ax = axes[2]
    for i, name in enumerate(names):
        scores = [p["similarity"] for p in models[name].get("pairs", []) if "similarity" in p]
        if scores:
            ax.scatter([i] * len(scores), scores, color=colors[i % len(colors)], s=60, zorder=3)
        else:
            ax.text(i, 0, "Unavailable", rotation=90, ha="center")
    ax.axhline(report["threshold"], color="#64748b", linestyle="--", label="Decision threshold")
    ax.set(title="Pair cosine similarity", ylabel="Cosine similarity", ylim=(-1.1, 1.15))
    ax.set_xticks(range(len(names)), names, rotation=20)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower left", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    same_image_only = all(p.get("image_a") == p.get("image_b")
                         for m in models.values() for p in m.get("pairs", []))
    note = ("Identical-image sanity check only; these scores do not measure recognition accuracy."
            if same_image_only else "Pair scores reflect the supplied dataset; interpret with its labels and coverage.")
    fig.text(0.5, 0.075, note, ha="center", fontsize=11)
    fig.text(0.5, 0.025, f"Provider: {report.get('provider', 'unknown')} | Warmed extraction timing; repeats needed for latency conclusions.\n"
             "Size includes all pack ONNX files, including unused auxiliary models. Failed results are not plotted as zero.",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output.with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Saved {output}.png and {output}.svg")


if __name__ == "__main__":
    main()

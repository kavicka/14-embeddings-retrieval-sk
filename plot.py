"""Turn results_*.json into figures. No measuring here."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

SYSTEMS = (
    ("bm25", "BM25", "#94a3b8"),
    ("bge-m3", "bge-m3", "#f97316"),
    ("inbatch", "fine-tuned", "#2563eb"),
    ("hardneg", "fine-tuned + hard negatives", "#7c3aed"),
)
STRATEGIES = (("fixed200", "~200 tokens"), ("fixed500", "~500 tokens"), ("article", "by article"))
FAMILIES = (
    ("hnswlib", "hnswlib", "#2563eb"),
    ("faiss_hnsw", "FAISS HNSW", "#7c3aed"),
    ("faiss_ivf", "FAISS IVF", "#059669"),
)


def retrieval() -> None:
    runs = json.load(open("results_retrieval.json"))["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, metric, title in zip(axes, ("recall@5", "mrr@10"), ("Recall@5", "MRR@10")):
        width = 0.2
        for i, (key, label, color) in enumerate(SYSTEMS):
            names = [f"{key}/{s}" for s, _ in STRATEGIES]
            values = [runs[n][metric] for n in names]
            errors = [
                [v - runs[n][f"{metric}_ci95"][0] for v, n in zip(values, names)],
                [runs[n][f"{metric}_ci95"][1] - v for v, n in zip(values, names)],
            ]
            ax.bar(
                [x + (i - 1.5) * width for x in range(len(STRATEGIES))],
                values,
                width,
                yerr=errors,
                capsize=2,
                color=color,
                label=label,
            )
        ax.set_xticks(range(len(STRATEGIES)), [t for _, t in STRATEGIES])
        ax.set(title=f"{title} on the golden questions (95 % interval)", ylim=(0, 1))
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig("assets/retrieval.png", dpi=150)


def ann() -> None:
    res = json.load(open("results_ann.json"))
    fig, ax = plt.subplots(figsize=(8.6, 5))
    for family, label, color in FAMILIES:
        groups = sorted({r.get("M", 0) for r in res["runs"].values() if r["family"] == family})
        for m in groups:
            points = sorted(
                (r["recall"], r["median_ms"])
                for r in res["runs"].values()
                if r["family"] == family and r.get("M", 0) == m
            )
            name = f"{label} M={m}" if m else label
            ax.plot(*zip(*points), "-o", ms=3, color=color, alpha=0.4 + 0.2 * groups.index(m), label=name)
    brute = res["runs"]["brute"]
    ax.axhline(brute["median_ms"], color="#dc2626", ls="--", label=f"own brute force, {brute['median_ms']:.1f} ms")
    ax.set(
        xlabel="recall@10 against exact search",
        ylabel="median latency per query (ms)",
        yscale="log",
        title=f"{res['vectors']:,} vectors, one query on one thread",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("assets/recall_latency.png", dpi=150)


def training() -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name, label, color in (
        ("inbatch", "in-batch negatives", "#2563eb"),
        ("hardneg", "+ hard negatives", "#7c3aed"),
    ):
        r = json.load(open(f"results_train_{name}.json"))
        ys = [r["val_base"]["mrr@10"]] + [h["val"]["mrr@10"] for h in r["history"]]
        ax.plot(range(len(ys)), ys, "-o", color=color, label=label)
    ax.set_xticks(range(len(ys)))
    ax.set(
        xlabel="epoch (0 = base model)",
        ylabel="MRR@10 on val documents",
        title="Val decides the model; test stays out of it",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("assets/training.png", dpi=150)


if __name__ == "__main__":
    Path("assets").mkdir(exist_ok=True)
    retrieval()
    training()
    ann()

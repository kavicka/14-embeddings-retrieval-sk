"""The visual example: one query descending through the layers of my own HNSW, on 2D points you can see.

Writes assets/hnsw_descent.gif and results_tiny_hnsw.json (recall of the tiny index against exact search).
"""

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from index import TinyHNSW

N, M, K, SEED = 1500, 6, 5, 14
COLORS = {
    "point": "#cbd5e1",
    "edge": "#e2e8f0",
    "at": "#2563eb",
    "looked": "#f97316",
    "query": "#dc2626",
    "found": "#059669",
}


def draw(index: TinyHNSW, q: np.ndarray, trace: list[dict], upto: int, found: list[int] | None) -> Image.Image:
    layers = len(index.graph)
    fig, axes = plt.subplots(1, layers, figsize=(3.1 * layers, 3.5))
    step = trace[min(upto, len(trace) - 1)]
    for layer, ax in zip(range(layers - 1, -1, -1), axes):  # top first: the eye follows the descent left to right
        nodes = list(index.graph[layer])
        pts = index.x[nodes]
        if layer > 0:
            for a in nodes:
                for b in index.graph[layer][a]:
                    ax.plot(*index.x[[a, b]].T, color=COLORS["edge"], lw=0.4, zorder=1)
        ax.scatter(*pts.T, s=2 if layer == 0 else 10, color=COLORS["point"], zorder=2)
        path = [t["at"] for t in trace[: upto + 1] if t["layer"] == layer]
        if path:
            ax.plot(*index.x[path].T, color=COLORS["at"], lw=1.2, zorder=3)
            ax.scatter(*index.x[path].T, s=12, color=COLORS["at"], zorder=4)
        if step["layer"] == layer and step["looked_at"]:
            ax.scatter(*index.x[step["looked_at"]].T, s=14, color=COLORS["looked"], zorder=4)
        ax.scatter(*q, marker="*", s=90, color=COLORS["query"], zorder=5)
        if found and layer == 0:
            ax.scatter(*index.x[found].T, s=22, color=COLORS["found"], zorder=6)
        ax.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(0, 1), title=f"layer {layer}: {len(nodes)} points")
    fig.suptitle("Blue: path so far · orange: neighbours checked now · star: query · green: answer", fontsize=9)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=80)
    plt.close(fig)
    return Image.open(buffer).convert("P", palette=Image.ADAPTIVE, colors=32)


def main() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.random((N, 2))
    index = TinyHNSW(x, m=M, seed=SEED)
    queries = rng.random((300, 2))
    truth = [np.argsort(((x - p) ** 2).sum(1))[:K] for p in queries]
    sweep = {
        ef: float(np.mean([len(set(index.search(p, K, ef)) & set(t)) / K for p, t in zip(queries, truth)]))
        for ef in (5, 10, 20, 40)
    }
    trace: list[dict] = []
    q = np.array([0.72, 0.31])
    found = index.search(q, K, 10, trace)
    frames = [draw(index, q, trace, i, None) for i in range(len(trace))]
    frames += [draw(index, q, trace, len(trace) - 1, found)] * 8
    Path("assets").mkdir(exist_ok=True)
    frames[0].save(
        "assets/hnsw_descent.gif", save_all=True, append_images=frames[1:], duration=350, loop=0, optimize=True
    )
    summary = {
        "points": N,
        "m": M,
        "k": K,
        "layers": [len(g) for g in index.graph],
        "recall_by_ef": sweep,
        "steps_in_gif": len(trace),
    }
    Path("results_tiny_hnsw.json").write_text(json.dumps(summary, indent=1))
    print(summary)


if __name__ == "__main__":
    main()

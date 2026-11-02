"""Before and after in 2D: do questions move next to their own passages?

PCA is fitted separately for each model, because each has its own space. The picture is only an illustration, so
the numbers come from the full 1024 dimensions: similarity to the own passage against similarity to the others.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from common import BASE_MODEL, docs_by_id, load_questions, save_json
from evaluate import encode, load_model

SAMPLE, SEED = 40, 14


def margin(q: np.ndarray, p: np.ndarray) -> dict:
    """Own passage minus the best other passage: above zero means this question would rank its passage first here."""
    sims = q @ p.T
    own = np.diag(sims)
    others = sims - np.eye(len(q)) * 2
    return {
        "mean_own": float(own.mean()),
        "mean_other": float((sims.sum() - own.sum()) / (len(q) * (len(q) - 1))),
        "share_own_is_best": float((own > others.max(1)).mean()),
    }


def main() -> None:
    chosen = json.load(open("results_best.json"))["best"]
    docs = docs_by_id("test")
    golden = load_questions("golden.json")
    picks = np.random.default_rng(SEED).choice(len(golden), SAMPLE, replace=False)
    questions = [golden[i]["question"] for i in picks]
    passages = [
        docs[golden[i]["doc_id"]]["text"][max(0, golden[i]["span"][0] - 200) : golden[i]["span"][1] + 200]
        for i in picks
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    result: dict = {"sample": SAMPLE, "chosen": chosen}
    for ax, (label, path) in zip(axes, (("base bge-m3", BASE_MODEL), ("fine-tuned", f"models/{chosen}"))):
        model = load_model(path)
        q, p = encode(model, questions), encode(model, passages)
        result[label] = margin(q, p)
        xy = PCA(2, random_state=SEED).fit_transform(np.concatenate([q, p]))
        a, b = xy[:SAMPLE], xy[SAMPLE:]
        for s, e in zip(a, b):
            ax.plot(*zip(s, e), color="#cbd5e1", lw=0.8, zorder=1)
        ax.scatter(*a.T, s=22, color="#2563eb", label="question", zorder=2)
        ax.scatter(*b.T, s=22, color="#f97316", marker="s", label="its passage", zorder=2)
        ax.set(
            xticks=[],
            yticks=[],
            title=f"{label}: own passage most similar for {result[label]['share_own_is_best']:.0%}",
        )
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("assets/projection.png", dpi=150)
    save_json("results_projection.json", result)
    print(result)


if __name__ == "__main__":
    main()

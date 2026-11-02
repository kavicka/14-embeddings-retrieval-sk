"""What changed for which questions: a paired comparison of base and fine-tuned on the same golden questions."""

import json
from collections import defaultdict

import numpy as np

from common import load_questions, save_json

STRATEGY = "fixed200"


def reciprocal(ranks: np.ndarray) -> np.ndarray:
    found = (ranks > 0) & (ranks <= 10)
    return found / np.where(found, ranks, 1)


def paired(a: np.ndarray, b: np.ndarray, seed: int = 0, rounds: int = 5000) -> dict:
    """Bootstrap the per-question difference: the same questions on both sides remove most of the noise."""
    rng = np.random.default_rng(seed)
    diff = b - a
    means = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(rounds)]
    return {
        "mean_difference": float(diff.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def in5(ranks: np.ndarray) -> np.ndarray:
    return ((ranks > 0) & (ranks <= 5)).astype(float)


def main() -> None:
    ranks = {k: np.array(v) for k, v in json.load(open("data/ranks.json")).items()}
    golden = load_questions("golden.json")
    chosen = json.load(open("results_best.json"))["best"]
    base, tuned = ranks[f"bge-m3/{STRATEGY}"], ranks[f"{chosen}/{STRATEGY}"]
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(golden):
        by_doc[q["doc_id"]].append(i)
    per_doc = {
        d: {
            "questions": len(ids),
            "base_recall@5": float(in5(base)[ids].mean()),
            "tuned_recall@5": float(in5(tuned)[ids].mean()),
        }
        for d, ids in by_doc.items()
    }
    gained = [i for i in range(len(golden)) if not in5(base)[i] and in5(tuned)[i]]
    lost = [i for i in range(len(golden)) if in5(base)[i] and not in5(tuned)[i]]
    missed = [i for i in range(len(golden)) if not in5(tuned)[i]]

    def show(ids: list[int]) -> list[dict]:
        return [
            {
                "question": golden[i]["question"],
                "doc_id": golden[i]["doc_id"],
                "base_rank": int(base[i]),
                "tuned_rank": int(tuned[i]),
            }
            for i in ids
        ]

    groups = {
        "rewritten_by_hand": [i for i, q in enumerate(golden) if q["rewritten"]],
        "as_generated": [i for i, q in enumerate(golden) if not q["rewritten"]],
        "short_under_80_chars": [i for i, q in enumerate(golden) if len(q["question"]) < 80],
        "long_80_chars_or_more": [i for i, q in enumerate(golden) if len(q["question"]) >= 80],
    }
    systems = {"base": base, "inbatch": ranks[f"inbatch/{STRATEGY}"], "hardneg": ranks[f"hardneg/{STRATEGY}"]}
    by_group = {
        g: {"questions": len(ids), **{n: float(in5(r)[ids].mean()) for n, r in systems.items()}}
        for g, ids in groups.items()
    }
    other = "inbatch" if chosen == "hardneg" else "hardneg"
    rival = ranks[f"{other}/{STRATEGY}"]
    result = {
        "chosen": chosen,
        "other": other,
        "other_vs_base_recall@5": paired(in5(base), in5(rival)),
        "other_vs_base_mrr@10": paired(reciprocal(base), reciprocal(rival)),
        "other_minus_chosen_recall@5": paired(in5(tuned), in5(rival)),
        "other_minus_chosen_mrr@10": paired(reciprocal(tuned), reciprocal(rival)),
        "strategy": STRATEGY,
        "recall@5_difference": paired(in5(base), in5(tuned)),
        "mrr@10_difference": paired(reciprocal(base), reciprocal(tuned)),
        "per_document": per_doc,
        "recall@5_by_group": by_group,
        "gained_top5": len(gained),
        "lost_top5": len(lost),
        "still_missed_top5": len(missed),
        "not_in_top10": int(((tuned == 0) | (tuned > 10)).sum()),
        "examples_gained": show(gained[:6]),
        "examples_lost": show(lost[:6]),
        "examples_missed": show(missed[:10]),
    }
    save_json("results_analysis.json", result)
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("examples")}, ensure_ascii=False)[:900])


if __name__ == "__main__":
    main()

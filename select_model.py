"""Choose the model on val documents only. The golden set is reported afterwards and never changes this choice."""

import json

from common import save_json


def main() -> None:
    best = {}
    for name in ("inbatch", "hardneg"):
        history = json.load(open(f"results_train_{name}.json"))["history"]
        top = max(history, key=lambda h: h["val"]["mrr@10"])
        best[name] = {"epoch": top["epoch"], "val_mrr@10": top["val"]["mrr@10"], "val_recall@5": top["val"]["recall@5"]}
    chosen = max(best, key=lambda n: best[n]["val_mrr@10"])
    save_json("results_best.json", {"rule": "highest val MRR@10 across epochs", "candidates": best, "best": chosen})
    print(chosen, best)


if __name__ == "__main__":
    main()

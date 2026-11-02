"""Fine-tune bge-m3 on question and passage pairs with in-batch negatives.

In a batch of 8 pairs, each question's own passage is the positive and the other 7 passages are negatives. The
second run adds a hard negative per pair: a passage the base model wrongly ranks high, and which lacks the evidence.
Each epoch is scored on the val documents, and the best epoch is kept. Test is never used here.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import losses
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from common import BASE_MODEL, Chunk, build_chunks, docs_by_id, first_hits, load_questions, save_json, scores, squeeze
from evaluate import dense_rankings, load_model
from make_questions import passages

HP = {
    "batch_size": 8,
    "epochs": 3,
    "lr": 2e-5,
    "warmup_steps": 20,
    "max_seq_length": 192,
    "temperature_scale": 20.0,
    "max_grad_norm": 1.0,
    "seed": 14,
    "hard_negative_pool": 10,
}


def load_pairs(split: str) -> list[dict]:
    docs = docs_by_id(split)
    rows = [r for r in load_questions(f"data/questions_{split}.jsonl") if r["status"] == "ok"]
    for r in rows:
        r["positive"] = docs[r["doc_id"]]["text"][r["passage"][0] : r["passage"][1]]
    return rows


def mine_hard_negatives(pairs: list[dict]) -> None:
    """What the untrained model confuses with the answer, in the same passage format as the positives.

    The first version mined 200-token chunks. Longer negatives against shorter positives taught "long means wrong",
    and val MRR fell to 0.19. A neighbour overlapping the positive is skipped too: it can hold part of the answer.
    """
    docs = docs_by_id("train")
    model = load_model(BASE_MODEL)
    pool = [Chunk(d["doc_id"], s, e, d["text"][s:e]) for d in docs.values() for s, e in passages(d["text"])]
    rankings = dense_rankings(model, pool, pairs)
    missing = 0
    for row, ranked in zip(pairs, rankings):
        evidence = squeeze(docs[row["doc_id"]]["text"][row["span"][0] : row["span"][1]])
        a, b = row["passage"]
        row["negative"] = next(
            (
                pool[j].text
                for j in ranked[: HP["hard_negative_pool"]]
                if not (pool[j].doc_id == row["doc_id"] and pool[j].start < b and a < pool[j].end)
                and evidence not in squeeze(pool[j].text)
            ),
            None,
        )
        missing += row["negative"] is None
    print("pairs without a hard negative:", missing, flush=True)
    del model


def val_score(model, chunks, questions, docs) -> dict:
    model.eval()
    model.max_seq_length = 512
    hits = first_hits(dense_rankings(model, chunks, questions), chunks, questions, docs)
    model.max_seq_length = HP["max_seq_length"]
    return scores(hits)


def train(name: str, hard: bool) -> None:
    random.seed(HP["seed"])
    torch.manual_seed(HP["seed"])
    pairs = load_pairs("train")
    if hard:
        mine_hard_negatives(pairs)
        pairs = [p for p in pairs if p["negative"]]
    val_docs = docs_by_id("val")
    val_questions = [r for r in load_questions("data/questions_val.jsonl") if r["status"] == "ok"]
    model = load_model(BASE_MODEL)
    val_chunks = build_chunks(list(val_docs.values()), model.tokenizer, "fixed200")
    baseline = val_score(model, val_chunks, val_questions, val_docs)
    print("val base", baseline, flush=True)
    model[
        0
    ].auto_model.gradient_checkpointing_enable()  # at batch 16 the GPU ran out of memory and macOS killed the run
    criterion = losses.MultipleNegativesRankingLoss(model, scale=HP["temperature_scale"])
    optimizer = AdamW(model.parameters(), lr=HP["lr"])
    steps = HP["epochs"] * (len(pairs) // HP["batch_size"])
    scheduler = LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / HP["warmup_steps"]) * max(0.0, 1 - s / steps))
    history, best, begun = [], -1.0, time.time()
    path = Path(f"models/{name}")
    for epoch in range(1, HP["epochs"] + 1):
        model.train()
        random.shuffle(pairs)
        running = []
        for i in range(0, len(pairs) - HP["batch_size"] + 1, HP["batch_size"]):
            batch = pairs[i : i + HP["batch_size"]]
            columns = [[p["question"] for p in batch], [p["positive"] for p in batch]]
            if hard:
                columns.append([p["negative"] for p in batch])
            features = [
                {k: v.to(model.device) for k, v in model.preprocess(c).items() if torch.is_tensor(v)} for c in columns
            ]
            loss = criterion(features, None)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            running.append(loss.item())
            if len(running) % 25 == 0:
                print(
                    name,
                    epoch,
                    len(running),
                    round(float(np.mean(running[-25:])), 4),
                    round(time.time() - begun),
                    "s",
                    flush=True,
                )
        val = val_score(model, val_chunks, val_questions, val_docs)
        history.append({"epoch": epoch, "loss": float(np.mean(running)), "val": val})
        print(name, history[-1], flush=True)
        if val["mrr@10"] > best:
            best = val["mrr@10"]
            model.save(str(path))
    result = {"name": name, "hard_negatives": hard, "pairs": len(pairs), "hyperparameters": HP, "val_base": baseline, "history": history, "seconds": time.time() - begun, "device": str(model.device)}  # fmt: skip
    save_json(f"results_train_{name}.json", result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--hard", action="store_true")
    a = ap.parse_args()
    if not Path(f"results_train_{a.name}.json").exists():
        train(a.name, a.hard)
    print(json.dumps(json.load(open(f"results_train_{a.name}.json"))["history"])[:400])

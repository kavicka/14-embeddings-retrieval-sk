"""Retrieval on a question set: BM25 and every embedding model, across chunking strategies.

Writes results_retrieval.json and, per system, the rank of the first relevant chunk for each question.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from common import (
    BASE_MODEL,
    build_chunks,
    docs_by_id,
    first_hits,
    load_questions,
    recall_ceiling,
    save_json,
    with_intervals,
)
from index import BruteForce

STRATEGIES = ("fixed200", "fixed500", "article")
TOP = 10
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_rankings(chunks, questions) -> np.ndarray:
    index = BM25Okapi([words(c.text) for c in chunks])
    return np.array([np.argsort(-index.get_scores(words(q["question"])))[:TOP] for q in questions])


def encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def dense_rankings(model, chunks, questions) -> np.ndarray:
    return BruteForce(encode(model, [c.text for c in chunks])).search(
        encode(model, [q["question"] for q in questions]), TOP
    )


def load_model(path: str) -> SentenceTransformer:
    model = SentenceTransformer(path, device=DEVICE)
    model.max_seq_length = 512
    return model


def main(models: dict[str, str], question_path: str, split: str, out: str) -> None:
    docs = docs_by_id(split)
    questions = load_questions(question_path)
    tokenizer = load_model(BASE_MODEL).tokenizer
    results: dict = (
        json.loads(Path(out).read_text())
        if Path(out).exists()
        else {"questions": len(questions), "runs": {}, "chunks": {}}
    )
    ranks: dict[str, list[int]] = (
        json.loads(Path("data/ranks.json").read_text()) if Path("data/ranks.json").exists() else {}
    )
    for strategy in STRATEGIES:
        chunks = build_chunks(list(docs.values()), tokenizer, strategy)
        results["chunks"][strategy] = {
            "count": len(chunks),
            "ceiling": recall_ceiling(chunks, questions, docs),
            "mean_chars": float(np.mean([len(c.text) for c in chunks])),
        }
        print(strategy, results["chunks"][strategy], flush=True)
        for name in ["bm25", *models]:
            key = f"{name}/{strategy}"
            if key in results["runs"]:
                continue
            rankings = (
                bm25_rankings(chunks, questions)
                if name == "bm25"
                else dense_rankings(load_model(models[name]), chunks, questions)
            )
            hits = first_hits(rankings, chunks, questions, docs)
            ranks[key] = hits.tolist()
            results["runs"][key] = with_intervals(hits)
            print(
                key,
                {
                    k: round(v, 3)
                    for k, v in results["runs"][key].items()
                    if k in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")
                },
                flush=True,
            )
            save_json(out, results)
            save_json("data/ranks.json", ranks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="name=path of a fine-tuned model")
    ap.add_argument("--questions", default="golden.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results_retrieval.json")
    a = ap.parse_args()
    main({"bge-m3": BASE_MODEL, **dict(m.split("=") for m in a.model)}, a.questions, a.split, a.out)

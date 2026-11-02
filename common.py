"""Chunking, relevance and metrics. All systems are judged here, so a chunking change cannot move the labels."""

import json
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from corpus import HEADING

BASE_MODEL = "BAAI/bge-m3"
MAX_SEQ = 512
COVERAGE = 0.8  # a chunk answers if it holds this share of the evidence: a cut through the last word is not a miss
BOOTSTRAP_SEED = 0


@dataclass
class Chunk:
    doc_id: str
    start: int
    end: int
    text: str


def load_docs(split: str) -> list[dict]:
    return [d for d in map(json.loads, open("data/docs.jsonl")) if d["split"] == split]


def squeeze(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def fixed_spans(offsets: list[tuple[int, int]], size: int, overlap: int) -> list[tuple[int, int]]:
    """Windows of `size` tokens. Character offsets come from the tokenizer, so a chunk can be matched to a span."""
    step = size - overlap
    return [
        (offsets[i][0], offsets[min(i + size, len(offsets)) - 1][1])
        for i in range(0, max(len(offsets) - overlap, 1), step)
    ]


def chunk_doc(doc: dict, tokenizer, strategy: str) -> list[Chunk]:
    text = doc["text"]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)
    offsets = [o for o in encoded["offset_mapping"] if o[1] > o[0]]
    if strategy == "fixed200":
        spans = fixed_spans(offsets, 200, 40)
    elif strategy == "fixed500":
        spans = fixed_spans(offsets, 500, 100)
    elif strategy == "article":
        spans = article_spans(text, offsets)
    else:
        raise ValueError(strategy)
    return [Chunk(doc["doc_id"], s, e, text[s:e]) for s, e in spans if e > s]


def article_spans(text: str, offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Cut at article headings. An article too long for the model falls back to fixed windows inside it."""
    cuts = [0] + [m.start() for m in HEADING.finditer(text)] + [len(text)]
    spans = []
    for a, b in pairwise(cuts):
        inside = [o for o in offsets if a <= o[0] and o[1] <= b]
        if not inside:
            continue
        spans += [(a, b)] if len(inside) <= MAX_SEQ else fixed_spans(inside, 500, 100)
    return spans


def build_chunks(docs: list[dict], tokenizer, strategy: str) -> list[Chunk]:
    return [c for d in docs for c in chunk_doc(d, tokenizer, strategy)]


def load_questions(path: str) -> list[dict]:
    return json.load(open(path)) if path.endswith(".json") else [json.loads(line) for line in open(path)]


def evidence_text(q: dict, docs: dict[str, dict]) -> str:
    return squeeze(docs[q["doc_id"]]["text"][q["span"][0] : q["span"][1]])


def relevant(chunk: Chunk, q: dict, evidence: str) -> bool:
    """Same place, or the same words: two documents of one insurer can hold the identical clause."""
    if chunk.doc_id == q["doc_id"]:
        s, e = q["span"]
        if max(0, min(chunk.end, e) - max(chunk.start, s)) >= COVERAGE * (e - s):
            return True
    return evidence in squeeze(chunk.text)


def first_hits(rankings: np.ndarray, chunks: list[Chunk], questions: list[dict], docs: dict[str, dict]) -> np.ndarray:
    """Rank (1-based) of the first relevant chunk per question, 0 when it is not in the list."""
    hits = np.zeros(len(questions), int)
    squeezed = [squeeze(c.text) for c in chunks]
    for i, q in enumerate(questions):
        ev = evidence_text(q, docs)
        s, e = q["span"]
        for rank, j in enumerate(rankings[i], 1):
            c = chunks[j]
            near = c.doc_id == q["doc_id"] and max(0, min(c.end, e) - max(c.start, s)) >= COVERAGE * (e - s)
            if near or ev in squeezed[j]:
                hits[i] = rank
                break
    return hits


def recall_ceiling(chunks: list[Chunk], questions: list[dict], docs: dict[str, dict]) -> float:
    """Share of questions any chunk can answer at all. A long span cut by a small chunk is lost before ranking starts."""
    squeezed = [squeeze(c.text) for c in chunks]
    ok = 0
    for q in questions:
        ev = evidence_text(q, docs)
        s, e = q["span"]
        ok += any(
            (c.doc_id == q["doc_id"] and max(0, min(c.end, e) - max(c.start, s)) >= COVERAGE * (e - s)) or ev in t
            for c, t in zip(chunks, squeezed)
        )
    return ok / len(questions)


def scores(hits: np.ndarray) -> dict[str, float]:
    """One information need per question, so nDCG reduces to the discount at the first relevant rank."""
    found = (hits > 0) & (hits <= 10)
    safe = np.where(found, hits, 1)
    return {
        "recall@1": float(np.mean(hits == 1)),
        "recall@5": float(np.mean((hits > 0) & (hits <= 5))),
        "recall@10": float(np.mean(found)),
        "mrr@10": float(np.mean(found / safe)),
        "ndcg@10": float(np.mean(found / np.log2(safe + 1))),
    }


def with_intervals(hits: np.ndarray, rounds: int = 2000) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = [scores(hits[rng.integers(0, len(hits), len(hits))]) for _ in range(rounds)]
    out = {"questions": len(hits)}
    for key, value in scores(hits).items():
        out[key] = value
        out[f"{key}_ci95"] = [float(np.percentile([b[key] for b in boots], p)) for p in (2.5, 97.5)]
    return out


def docs_by_id(split: str) -> dict[str, dict]:
    return {d["doc_id"]: d for d in load_docs(split)}


def save_json(path: str, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=1, ensure_ascii=False))

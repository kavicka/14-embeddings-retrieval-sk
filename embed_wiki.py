"""Embed Slovak Wikipedia passages once, for the ANN benchmark. Saved in shards, so a stopped run resumes.

The insurance corpus is far too small to need an approximate index, so speed is measured on 200,000 passages.
Queries are held-out passages from the validation text, never in the index.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import BASE_MODEL

SOURCE = "../12-gpt-from-scratch-sk/data/sk_train.txt"
QUERIES = "../12-gpt-from-scratch-sk/data/sk_val.txt"
TARGET, QUERY_COUNT, SHARD, MIN_CHARS, MAX_CHARS, SEED = 200_000, 2_000, 20_000, 200, 600, 14
OUT = Path("data/wiki")


def passages(path: str, limit: int, seed: int) -> list[str]:
    """Reservoir sampling: every paragraph has the same chance, and the 300 MB file never has to fit in a list."""
    rng, pick, seen = random.Random(seed), [], 0
    with open(path) as source:
        for line in source:
            line = line.strip()
            if len(line) < MIN_CHARS:
                continue
            seen += 1
            if len(pick) < limit:
                pick.append(line[:MAX_CHARS])
            elif (j := rng.randrange(seen)) < limit:
                pick[j] = line[:MAX_CHARS]
    return pick


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(BASE_MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
    model.max_seq_length = 128  # a benchmark of the index, not of the text: short passages keep this affordable
    model.half()
    if not (OUT / "passages.json").exists():
        (OUT / "passages.json").write_text(json.dumps(passages(SOURCE, TARGET, SEED), ensure_ascii=False))
        (OUT / "queries.json").write_text(json.dumps(passages(QUERIES, QUERY_COUNT, SEED + 1), ensure_ascii=False))
    texts = json.loads((OUT / "passages.json").read_text())
    begun = time.time()
    for k in range(0, len(texts), SHARD):
        path = OUT / f"shard_{k // SHARD:03d}.npy"
        if path.exists():
            continue
        vectors = model.encode(
            texts[k : k + SHARD],
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        np.save(path, vectors.astype(np.float32))
        print(k + SHARD, round(time.time() - begun), "s", flush=True)
    query_path = OUT / "queries.npy"
    if not query_path.exists():
        queries = json.loads((OUT / "queries.json").read_text())
        np.save(
            query_path,
            model.encode(queries, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32),
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()

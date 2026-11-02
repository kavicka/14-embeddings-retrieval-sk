"""Brute force vs approximate search on 200,000 bge-m3 vectors: recall against the exact answer, latency, build time, size.

Latency is one query at a time on one thread, because that is how a request meets an index. Each configuration is
saved when done, so a stopped run resumes. Run it on an idle machine: anything else on the CPU shows up as latency.
"""

import json
import time
from pathlib import Path

import faiss
import hnswlib
import numpy as np
from threadpoolctl import threadpool_limits

from common import save_json
from index import BruteForce

K, LATENCY_QUERIES = 10, 500
OUT = "results_ann.json"
HNSW_M = (8, 16, 32)
HNSW_EF = (10, 20, 50, 100, 200, 400)
EF_CONSTRUCTION = 200
IVF_LISTS, IVF_PROBES = 1024, (1, 4, 16, 64, 128)


def load_vectors() -> tuple[np.ndarray, np.ndarray]:
    shards = sorted(Path("data/wiki").glob("shard_*.npy"))
    return np.concatenate([np.load(s) for s in shards]), np.load("data/wiki/queries.npy")


def recall(found: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([len(set(a) & set(b)) / K for a, b in zip(found, truth)]))


def timed(search, queries: np.ndarray) -> dict:
    with threadpool_limits(1):
        times = []
        for q in queries[:LATENCY_QUERIES]:
            start = time.perf_counter()
            search(q[None])
            times.append(time.perf_counter() - start)
    ms = np.array(times) * 1000
    return {"median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95))}


def main() -> None:
    x, queries = load_vectors()
    print("vectors", x.shape, "queries", queries.shape, flush=True)
    res = json.loads(Path(OUT).read_text()) if Path(OUT).exists() else {"vectors": len(x), "dim": x.shape[1], "queries": len(queries), "k": K, "latency_queries": LATENCY_QUERIES, "runs": {}}  # fmt: skip
    brute = BruteForce(x)
    truth_path = Path("data/wiki/truth.npy")
    if truth_path.exists():
        truth = np.load(truth_path)
    else:
        truth = np.concatenate([brute.search(queries[i : i + 200], K) for i in range(0, len(queries), 200)])
        np.save(truth_path, truth)
    if "brute" not in res["runs"]:
        res["runs"]["brute"] = {
            "family": "brute",
            "recall": 1.0,
            "build_s": 0.0,
            "bytes": int(x.nbytes),
            **timed(lambda q: brute.search(q, K), queries),
        }
        save_json(OUT, res)
    print(res["runs"]["brute"], flush=True)
    for m in HNSW_M:
        if all(f"hnswlib_M{m}_ef{ef}" in res["runs"] for ef in HNSW_EF):
            continue
        begun = time.perf_counter()
        index = hnswlib.Index(space="ip", dim=x.shape[1])
        index.init_index(max_elements=len(x), M=m, ef_construction=EF_CONSTRUCTION, random_seed=14)
        index.add_items(x, num_threads=8)
        build = time.perf_counter() - begun
        path = Path(f"data/wiki/h{m}.bin")
        index.save_index(str(path))
        size = path.stat().st_size
        path.unlink()
        index.set_num_threads(1)
        for ef in HNSW_EF:
            index.set_ef(ef)
            found = index.knn_query(queries, k=K)[0]
            run = {
                "family": "hnswlib",
                "M": m,
                "ef": ef,
                "recall": recall(found, truth),
                "build_s": build,
                "bytes": size,
                **timed(lambda q, i=index: i.knn_query(q, k=K), queries),
            }
            res["runs"][f"hnswlib_M{m}_ef{ef}"] = run
            print(
                f"M{m} ef{ef}",
                {k: round(v, 4) for k, v in run.items() if k in ("recall", "median_ms", "p95_ms")},
                flush=True,
            )
        save_json(OUT, res)
    if "faiss_ivf_p1" not in res["runs"]:
        begun = time.perf_counter()
        quantizer = faiss.IndexFlatIP(x.shape[1])
        index = faiss.IndexIVFFlat(quantizer, x.shape[1], IVF_LISTS, faiss.METRIC_INNER_PRODUCT)
        index.train(x)
        index.add(x)
        build = time.perf_counter() - begun
        size = int(faiss.serialize_index(index).size)
        faiss.omp_set_num_threads(1)
        for p in IVF_PROBES:
            index.nprobe = p
            found = index.search(queries, K)[1]
            res["runs"][f"faiss_ivf_p{p}"] = {
                "family": "faiss_ivf",
                "nprobe": p,
                "recall": recall(found, truth),
                "build_s": build,
                "bytes": size,
                **timed(lambda q, i=index: i.search(q, K), queries),
            }
            print("ivf", p, round(res["runs"][f"faiss_ivf_p{p}"]["recall"], 4), flush=True)
        save_json(OUT, res)
    if "faiss_hnsw_ef10" not in res["runs"]:
        faiss.omp_set_num_threads(8)
        begun = time.perf_counter()
        index = faiss.IndexHNSWFlat(x.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = EF_CONSTRUCTION
        index.add(x)
        build = time.perf_counter() - begun
        size = int(faiss.serialize_index(index).size)
        faiss.omp_set_num_threads(1)
        for ef in HNSW_EF:
            index.hnsw.efSearch = ef
            found = index.search(queries, K)[1]
            res["runs"][f"faiss_hnsw_ef{ef}"] = {
                "family": "faiss_hnsw",
                "M": 16,
                "ef": ef,
                "recall": recall(found, truth),
                "build_s": build,
                "bytes": size,
                **timed(lambda q, i=index: i.search(q, K), queries),
            }
        save_json(OUT, res)
    print("ANN DONE", flush=True)


if __name__ == "__main__":
    main()

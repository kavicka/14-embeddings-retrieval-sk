"""My own vector search: exact brute force and a small HNSW, both on NumPy, for learning.

The production benchmark (`ann_benchmark.py`) compares `BruteForce` with hnswlib and FAISS on 200k vectors.
"""

import heapq

import numpy as np


class BruteForce:
    """Exact search. Vectors are normalised, so cosine similarity is just a dot product."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        sims = queries @ self.vectors.T
        top = np.argpartition(-sims, min(k, sims.shape[1] - 1), axis=1)[:, :k]  # O(n), a full sort is not needed
        order = np.take_along_axis(sims, top, 1).argsort(axis=1)[:, ::-1]
        return np.take_along_axis(top, order, 1)


class TinyHNSW:
    """Hierarchical navigable small world: a layered graph where upper layers are sparse highways.

    Search starts at the top, walks greedily towards the query on each sparse layer, and only the bottom layer,
    which holds every point, gets a wider search. `trace` keeps every step so the GIF can show the descent.
    """

    def __init__(self, vectors: np.ndarray, m: int = 8, ef_construction: int = 40, seed: int = 0):
        self.x = vectors
        self.m = m
        self.ml = 1 / np.log(m)  # each layer holds about 1/m of the one below
        self.rng = np.random.default_rng(seed)
        self.graph: list[dict[int, list[int]]] = []
        self.entry = -1
        for i in range(len(vectors)):
            self._insert(i, ef_construction)

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _greedy(self, q: np.ndarray, start: int, layer: int, trace: list | None = None) -> int:
        """Move to whichever neighbour is closer to the query, until none is: a local minimum on this layer."""
        here, best = start, self._dist(q, self.x[start])
        while True:
            neighbours = self.graph[layer].get(here, [])
            if trace is not None:
                trace.append({"layer": layer, "at": here, "looked_at": list(neighbours)})
            step = min(neighbours, key=lambda n: self._dist(q, self.x[n]), default=here)
            d = self._dist(q, self.x[step])
            if d >= best:
                return here
            here, best = step, d

    def _search_layer(
        self, q: np.ndarray, start: int, layer: int, ef: int, trace: list | None = None
    ) -> list[tuple[float, int]]:
        """Best-first search holding the ef closest found so far. Wider ef costs time and buys recall."""
        d0 = self._dist(q, self.x[start])
        seen, frontier, best = {start}, [(d0, start)], [(-d0, start)]
        while frontier:
            d, node = heapq.heappop(frontier)
            if d > -best[0][0] and len(best) >= ef:
                break
            neighbours = [n for n in self.graph[layer].get(node, []) if n not in seen]
            if trace is not None:
                trace.append({"layer": layer, "at": node, "looked_at": neighbours})
            for n in neighbours:
                seen.add(n)
                dn = self._dist(q, self.x[n])
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(frontier, (dn, n))
                    heapq.heappush(best, (-dn, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-d, n) for d, n in best)

    def _insert(self, i: int, ef: int) -> None:
        level = int(-np.log(1 - self.rng.random()) * self.ml)
        while len(self.graph) <= level:
            self.graph.append({})
        if self.entry < 0:
            self.entry = i
            for layer in range(level + 1):
                self.graph[layer][i] = []
            return
        q, at = self.x[i], self.entry
        for layer in range(len(self.graph) - 1, level, -1):
            if at in self.graph[layer]:
                at = self._greedy(q, at, layer)
        for layer in range(min(level, len(self.graph) - 1), -1, -1):
            if at not in self.graph[layer]:
                continue
            found = self._search_layer(q, at, layer, ef)
            limit = self.m * 2 if layer == 0 else self.m  # the bottom layer carries the real search, so more links
            self.graph[layer][i] = [n for _, n in found[: self.m]]
            for n in self.graph[layer][i]:
                links = self.graph[layer][n] + [i]
                links.sort(key=lambda o: self._dist(self.x[n], self.x[o]))
                self.graph[layer][n] = links[:limit]
            at = found[0][1]
        for layer in range(level + 1):
            self.graph[layer].setdefault(i, [])
        if level >= self._top_of_entry():
            self.entry = i

    def _top_of_entry(self) -> int:
        return max(layer for layer in range(len(self.graph)) if self.entry in self.graph[layer])

    def search(self, q: np.ndarray, k: int, ef: int = 40, trace: list | None = None) -> list[int]:
        at = self.entry
        for layer in range(self._top_of_entry(), 0, -1):
            at = self._greedy(q, at, layer, trace)
        return [n for _, n in self._search_layer(q, at, 0, max(ef, k), trace)[:k]]

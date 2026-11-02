"""Customer questions with evidence spans. A label is (doc_id, start, end) in the document text, never a chunk id,
so every chunking strategy is scored against the same answers.

Passages are sampled first, then a local model writes a question and quotes the answer. A quote that cannot be found
in the passage is dropped. Golden questions come from test documents and are then read and fixed one by one.
"""

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

WRITERS = {
    "golden": "gemma3:12b",
    "train": "qwen3:30b-a3b-instruct-2507-q4_K_M",
    "val": "qwen3:30b-a3b-instruct-2507-q4_K_M",
}
SOURCE_SPLIT = {"golden": "test", "train": "train", "val": "val"}
SIZES = {"golden": 220, "train": 2000, "val": 300}
SEEDS = {"golden": 14, "train": 15, "val": 16}
PROMPT = """Tu je úryvok z poistných podmienok ({product}):

\"\"\"{passage}\"\"\"

Predstav si bežného klienta, ktorý nepozná poistnú terminológiu. Napíš jednu otázku, ktorú by mohol položiť a na ktorú tento úryvok priamo odpovedá.
- Otázka je napísaná jeho vlastnými slovami, opisuje konkrétnu situáciu a nepreberá formulácie z úryvku.
- Musí dávať zmysel sama osebe: žiadne "v tomto článku" ani "podľa textu". Názov poisťovne nespomínaj.
Potom doslova skopíruj z úryvku jednu alebo dve vety, ktoré na otázku odpovedajú.
Ak v úryvku nie je nič, na čo by sa klient pýtal, vráť {{"question": null}}.
Odpovedz iba JSON objektom {{"question": "...", "evidence": "..."}}."""


def passages(text: str, low: int = 300, high: int = 900) -> list[tuple[int, int]]:
    """Runs of whole paragraphs between low and high characters: long enough to hold an answer, short enough to focus."""
    spans, start, end = [], None, 0
    for m in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text):
        for line in re.finditer(r"[^\n]+", m.group(0)):
            s, e = m.start() + line.start(), m.start() + line.end()
            start = s if start is None else start
            end = e
            if end - start >= low and text[end - 1] in ".;:":
                spans.append((start, end))
                start = None
            elif end - start > high:
                start = None
        start = None
    return spans


def locate(quote: str, text: str, start: int, end: int) -> tuple[int, int] | None:
    """Find the quote inside the passage, letters and digits only. PDF extraction leaves "b ola" and "pre -\\nskúmať"
    in the source while the model writes the clean word, so exact matching would reject true quotes."""
    kept = [(i, c.lower()) for i, c in enumerate(text[start:end]) if c.isalnum()]
    hay = "".join(c for _, c in kept)
    needle = "".join(c.lower() for c in quote if c.isalnum())
    if len(needle) < 20:
        return None
    at = hay.find(needle)
    if at < 0:
        return None
    return start + kept[at][0], start + kept[at + len(needle) - 1][0] + 1


def ask(model: str, prompt: str, seed: int) -> dict:
    body = {"model": model, "prompt": prompt, "stream": False, "format": "json", "think": False, "keep_alive": "10m", "options": {"temperature": 0.7, "seed": seed, "num_predict": 300}}  # fmt: skip
    req = urllib.request.Request(
        "http://localhost:11434/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    answer = json.load(urllib.request.urlopen(req, timeout=300))["response"]
    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        return {}


def main(name: str) -> None:
    docs = [json.loads(line) for line in open("data/docs.jsonl")]
    docs = [d for d in docs if d["split"] == SOURCE_SPLIT[name]]
    rng = random.Random(SEEDS[name])
    # every document gets its share, so one long PDF does not supply most of the questions
    pools = {d["doc_id"]: rng.sample(passages(d["text"]), len(passages(d["text"]))) for d in docs}
    order = []
    while len(order) < SIZES[name] and any(pools.values()):
        for d in docs:
            if pools[d["doc_id"]]:
                order.append((d, pools[d["doc_id"]].pop()))
    order = order[: SIZES[name]]
    path = Path(f"data/questions_{name}.jsonl")
    done = sum(1 for _ in open(path)) if path.exists() else 0
    with open(path, "a") as out:
        for i, (doc, (start, end)) in enumerate(order[done:], done):
            passage = doc["text"][start:end]
            reply = ask(WRITERS[name], PROMPT.format(product=doc["product"], passage=passage), SEEDS[name] + i)
            quote, question = reply.get("evidence"), reply.get("question")
            span = locate(quote, doc["text"], start, end) if isinstance(quote, str) else None
            status = "no_question" if not question else "quote_not_found" if span is None else "ok"
            row = {
                "i": i,
                "doc_id": doc["doc_id"],
                "passage": [start, end],
                "question": question,
                "span": span,
                "status": status,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0:
                print(name, i, status, flush=True)


def second_pass(name: str) -> None:
    """Re-ask every failed passage with the better matcher. Train also gets one new question per good passage,
    because the corpus has fewer passages than the pairs training wants."""
    docs = {d["doc_id"]: d for d in map(json.loads, open("data/docs.jsonl"))}
    path = Path(f"data/questions_{name}.jsonl")
    rows = [json.loads(line) for line in open(path)]
    first = [r for r in rows if not r.get("second")]
    done = {r["i"] for r in rows if r.get("second")}
    todo = [r for r in first if r["status"] == "quote_not_found" or (name == "train" and r["status"] == "ok")]
    with open(path, "a") as out:
        for n, row in enumerate(t for t in todo if t["i"] not in done):
            doc, (start, end) = docs[row["doc_id"]], row["passage"]
            prompt = PROMPT.format(product=doc["product"], passage=doc["text"][start:end])
            if row["status"] == "ok":
                prompt += f"\nNepýtaj sa na to isté ako táto otázka: {row['question']}"
            reply = ask(WRITERS[name], prompt, SEEDS[name] + 5000 + row["i"])
            quote, question = reply.get("evidence"), reply.get("question")
            span = locate(quote, doc["text"], start, end) if isinstance(quote, str) else None
            status = "no_question" if not question else "quote_not_found" if span is None else "ok"
            new = {
                "i": row["i"],
                "second": True,
                "doc_id": row["doc_id"],
                "passage": [start, end],
                "question": question,
                "quote": quote,
                "span": span,
                "status": status,
            }
            out.write(json.dumps(new, ensure_ascii=False) + "\n")
            out.flush()
            if n % 25 == 0:
                print(name, n, status, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(WRITERS))
    ap.add_argument("--second", action="store_true")
    a = ap.parse_args()
    if a.second:
        second_pass(a.name)
    else:
        main(a.name)

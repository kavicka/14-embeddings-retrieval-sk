# Tutorial · 14 · Embeddings & retrieval on Slovak insurance terms

🇸🇰 [Slovenská verzia](TUTORIAL.sk.md) · 📄 [README (results)](README.md)

This tutorial builds the project in the order you would write it: get real documents → split them before anything else → label answers as evidence spans → chunk and score in one place → baselines → fine-tune with in-batch negatives → hard negatives → choose on val → your own index → an approximate index at scale. Each step explains **what** the code does, **why** it is written that way, and **how** it works. The README has all the numbers.

**Goal:** given a customer's Slovak question, find the clause in real insurance terms that answers it, measure how well that works, and find out whether fine-tuning the embedding model and an approximate index are worth it.

**You need:** project 10 (norms, cosine similarity: why a dot product of unit vectors ranks like cosine), project 12 (its Slovak Wikipedia text, used only for the speed benchmark) and Ollama for writing questions. Libraries: `sentence-transformers` (bge-m3), `rank_bm25`, `hnswlib`, `faiss-cpu`, `pypdf`.

---

## Step 0 · What retrieval is, in one paragraph

An embedding model turns a text into a vector, here 1,024 numbers, so that texts with similar meaning land close together. Retrieval embeds every chunk of the documents once, embeds the question, and returns the chunks whose vectors are closest. With vectors normalised to length 1, "closest" is simply the largest dot product (project 10). Two things decide quality: **the model** (does it put "ukradli mi auto" near "odcudzenie vozidla"?) and **the chunks** (is the answer inside one chunk at all?). Speed is a third, separate question: comparing the question with every vector is exact but grows with the corpus; an approximate index trades a little recall for a lot of speed.

---

## Step 1 · Real documents, cleaned but with their structure ([`corpus.py`](corpus.py))

```python
SOURCES = [("slsp_majetok", "Slovenská sporiteľňa", "majetok", "https://…pdf"), …]   # 20 public PDFs
SPLITS = {"test": {"Generali", "UNIQA", "Union"}, "val": {"Wüstenrot", "mBank"}}
```

**What:** downloads 20 public general insurance terms (VPP) from 12 sources, extracts text with `pypdf`, cleans it and writes `data/docs.jsonl`. The PDFs and text stay in `data/` and are never committed; the URLs are in the code, so anyone can rebuild the corpus.

**Why these documents:** P7 builds a RAG system on exactly this kind of text, so the retrieval lessons here transfer directly. Legal Slovak is also a hard case: long sentences, inflection, and wording customers never use.

### Cleaning PDF text
```python
text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # join words hyphenated across lines
BROKEN_FONT = str.maketrans({"è": "č", "ò": "ň", "¾": "ľ", …})   # a font map some PDFs lost
```
- A line repeated on many pages is a header or footer, and a line of dots is the table of contents; `extract` drops both.
- Paragraph breaks are **kept**, because they carry the structure: articles, numbered points.
- `quality()` counts diacritics and common Slovak words per document. Garbage from a column layout or a broken font shows up immediately as almost no "č, š, ž" and no "poistenie".

**Important:** `data/docs.jsonl` must not change after labelling. Every label is a character offset into this text (Step 3), so re-running the cleaning with a different rule would silently move every answer.

---

## Step 2 · Split by insurer, before any question exists

```python
def split_of(insurer: str) -> str:
    return next((name for name, group in SPLITS.items() if insurer in group), "train")
```

**What:** whole insurers go to one side: 11 documents train, 3 val (Wüstenrot, mBank), 6 test (Generali, UNIQA, Union).

**Why by insurer, not by document or by chunk:** one insurer reuses its own boilerplate across products. If Generali's motor terms were in train and Generali's property terms in test, the model would have seen half the test wording already. Splitting by insurer is the realistic case too: you fine-tune on the documents you have and then meet a new insurer.

### Measuring the leak that remains
```python
def leakage(docs):   # share of a held-out document's 8-word runs that also occur in train
```
Even across insurers, legal phrases repeat ("poistník je povinný bez zbytočného odkladu…"). Between 1.6 % and 11.1 % of each held-out document's 8-word runs also appear in train. That is small enough to call the test unseen, and it is reported instead of hidden.

---

## Step 3 · Label answers as evidence spans, not chunk ids ([`make_questions.py`](make_questions.py))

```python
row = {"doc_id": doc["doc_id"], "passage": [start, end], "question": question, "span": span}
```

**What:** a label is a document id plus the character offsets of the sentence that answers. It says nothing about chunks.

**Why:** you will want to compare chunk sizes (Step 5). If a label were "chunk 412", every new chunking would need new labels, done by hand again. With a span, any chunking can be scored automatically: a chunk is relevant if it covers the span.

### How a question is made
1. `passages()` cuts each document into runs of whole paragraphs, 300–900 characters: long enough to hold an answer, short enough to keep the model focused.
2. Every document gets its share of passages in round-robin, so one 60-page PDF does not supply most of the questions.
3. A local model gets the passage and the instruction: imagine a customer who does not know the terminology, ask in your own words, describe a situation, do not copy phrases; then **quote** the sentence that answers.
4. `locate()` searches the quote inside the passage. If it is not there, the question is dropped: the model invented an answer.

### The matcher that lost 40 % of the questions
```python
kept = [(i, c.lower()) for i, c in enumerate(text[start:end]) if c.isalnum()]
needle = "".join(c.lower() for c in quote if c.isalnum())
```
The first version matched quotes exactly and rejected 40 % of them. The model was not lying: PDF extraction left "b ola" and "pre -\nskúmať" in the source, while the model wrote the clean word. Matching on letters and digits only, then mapping the hit back to original offsets, fixed it. `second_pass()` re-asked the lost passages with the better matcher.

### Different writers, on purpose
```python
WRITERS = {"golden": "gemma3:12b", "train": "qwen3:30b-a3b…", "val": "qwen3:30b-a3b…"}
```
The golden (test) questions come from a different model than training questions, so the fine-tuned model cannot win by recognising one writer's style. Remember that val shares train's writer: that detail decides Step 9.

---

## Step 4 · Read the golden set by hand ([`review_golden.py`](review_golden.py))

```python
DROPPED = {"evidence_does_not_answer": [24, 26, …], "fragment_too_short_to_judge": […], …}
REWRITTEN = {6: "Kedy presne začne platiť moje poistenie auta, keď podpíšem zmluvu?", …}
```

**What:** all 190 golden candidates were read one by one. 49 were dropped (16 unnatural or off-product, 13 where the evidence did not answer, 10 fragments, 8 duplicates, 2 needing the document open) and 20 were reworded. 141 remain in `golden.json`.

**Why the decisions live in code:** anyone can see exactly which question was dropped and why, and re-running the script reproduces the golden set. The span of a rewritten question stays as generated, so the answer location is still machine-checked.

**Why only the golden set is read in full:** it decides every number in the README. Training pairs tolerate some noise; for them every 10th was read (140 pairs, 11 with loosely matching evidence, kept).

---

## Step 5 · Chunking and scoring in one file ([`common.py`](common.py))

### Three chunking strategies
```python
if strategy == "fixed200":   spans = fixed_spans(offsets, 200, 40)     # 200 tokens, 40 overlap
elif strategy == "fixed500": spans = fixed_spans(offsets, 500, 100)
elif strategy == "article":  spans = article_spans(text, offsets)      # cut at "Článok 5 …"
```
- Sizes are counted in **tokens of the model's own tokenizer**, and `return_offsets_mapping` gives each token's character position, so a chunk knows exactly which characters it covers.
- Overlap means an answer cut by a boundary still fits whole into the next chunk.
- `article_spans` cuts at headings; an article longer than the model's 512-token limit falls back to fixed windows inside it, because bge-m3 would silently ignore the rest.

### When does a chunk answer?
```python
def relevant(chunk, q, evidence):
    if chunk.doc_id == q["doc_id"]:
        if overlap(chunk, q["span"]) >= COVERAGE * span_length:   # COVERAGE = 0.8
            return True
    return evidence in squeeze(chunk.text)
```
- 80 % coverage, not 100 %: a chunk boundary through the last word of the answer is not a miss.
- **Or the same words anywhere:** one insurer can have the identical clause in two documents, and finding the other copy is a correct answer.
- `recall_ceiling` reports what share of questions any chunk can answer at all. With 200-token chunks it is 98.6 %: a few long answers are cut in two before ranking even starts.

### The metrics
```python
"recall@5": mean((hits > 0) & (hits <= 5)),      # the answer is somewhere in the top 5
"mrr@10":   mean(found / rank),                   # 1 for rank 1, 0.5 for rank 2, …
"ndcg@10":  mean(found / log2(rank + 1)),         # a softer discount than MRR
```
`hits` holds, per question, the rank of the first relevant chunk (0 = not in the top 10). With one answer per question, nDCG reduces to the discount at that rank. `with_intervals` bootstraps over questions: resample the 141 questions 2,000 times and take the 2.5th and 97.5th percentile. With 141 questions those intervals are wide, and that matters in Step 9.

**Why everything is in one file:** BM25, the base model and both fine-tuned models are scored by the same `first_hits` and `scores`. No system can win through a friendlier scorer.

---

## Step 6 · Baselines: BM25 and bge-m3 ([`evaluate.py`](evaluate.py))

```python
def bm25_rankings(chunks, questions):
    index = BM25Okapi([words(c.text) for c in chunks])
def dense_rankings(model, chunks, questions):
    return BruteForce(encode(model, chunk_texts)).search(encode(model, questions), TOP)
```

**BM25** scores word overlap, weighted so rare words count more. It gets recall@5 **0.191**. Customers write "ukradli" where the terms say "odcudzenie", and Slovak inflection turns one word into many forms; without stemming, BM25 sees no match.

**bge-m3**, a multilingual embedding model, gets **0.574** without any training. `normalize_embeddings=True` makes every vector length 1, so the dot product in `BruteForce` is cosine similarity.

**Chunking** barely moves bge-m3 (0.574 / 0.546 / 0.567, inside the noise), while BM25 improves with longer chunks: more words, more chances to overlap.

---

## Step 7 · Fine-tuning with in-batch negatives ([`train_embed.py`](train_embed.py))

```python
columns = [[p["question"] for p in batch], [p["positive"] for p in batch]]
loss = losses.MultipleNegativesRankingLoss(model, scale=20.0)(features, None)
```

**How the loss works:** take a batch of 8 (question, passage) pairs. Embed all 8 questions and all 8 passages, and compute the 8×8 matrix of similarities. Row *i* should be highest in column *i*, its own passage; the other 7 passages in the batch are free negatives. The loss is cross-entropy over each row (project 08), with similarities multiplied by `scale` 20 so the softmax is sharp enough to learn from. This is contrastive learning: pull matching pairs together, push the rest apart.

**Why batch size matters, and why it is 8:** a bigger batch means more negatives per question, usually better. Batch 16 ran out of GPU memory and macOS killed the process **without a Python error**: a silent death can be memory. Gradient checkpointing (`gradient_checkpointing_enable()`) recomputes activations in the backward pass instead of storing them, trading time for memory.

**The other hyperparameters:** learning rate 2e-5 (small: the model is already good and should move, not forget), warmup 20 steps then linear decay, 3 epochs, gradients clipped at 1.0, seed 14, all saved in `results_train_*.json`.

**Choosing the epoch:** after each epoch, `val_score` runs retrieval on the val documents, and the model is saved only if val MRR@10 improved. Test documents are never touched here.

Result on the golden set: recall@5 0.574 → **0.674**, MRR@10 0.392 → 0.456.

---

## Step 8 · Hard negatives, and a bug worth keeping

```python
row["negative"] = next(pool[j].text for j in ranked[:10]
                       if not overlaps_positive(pool[j]) and evidence not in squeeze(pool[j].text))
columns.append([p["negative"] for p in batch])
```

**What:** random in-batch negatives are often easy (a travel clause vs a car question). A hard negative is a passage the **base model** wrongly ranks high for the question, so training focuses on the confusions that actually happen. `mine_hard_negatives` takes the top 10 for each training question and picks the first that is neither the positive nor an overlapping neighbour, nor contains the evidence.

**The bug:** the first version mined 200-token chunks while positives were 300–900-character passages, and it did not skip neighbours overlapping the answer. Val MRR fell from 0.433 to 0.19–0.28, and recall was worst on long chunks. The model most likely learned "the longer text is the wrong one". Negatives teach whatever differs from the positives; if that is length, the model learns length. Mining in the same passage format and skipping overlaps fixed it. The failed run is kept in `results_train_hardneg_chunks.json`.

---

## Step 9 · Choose on val, report on test ([`select_model.py`](select_model.py))

```python
chosen = max(best, key=lambda n: best[n]["val_mrr@10"])
```

**The rule, fixed before the golden numbers existed:** the model with the highest val MRR@10. Hard negatives won on val (0.610 vs 0.569).

**What test said:** in-batch 0.674 recall@5, hard negatives 0.617. Val picked the weaker model, and the chosen one shows no provable gain over base (+0.043, interval −0.071 to 0.142).

**Why it was not switched:** picking after seeing test turns the test set into a second val set, and the reported number stops meaning anything. Both rows stay in the table, with the explanation.

**Why val failed** ([`analysis.py`](analysis.py)): splitting the golden questions by length shows it. Under 80 characters, base 0.634 vs tuned 0.537 / 0.488; 80 characters or more, base 0.550 vs tuned 0.730 / 0.670. Training questions averaged 144 characters, golden 108. Val came from the same writer as train (141 characters), so it rewarded exactly the bias that hurt on test. **Selection data should differ from training data the way production does.**

`analysis.py` also bootstraps **paired** differences: the same question on both sides, so per-question difficulty cancels out and the interval is narrower than comparing two separate intervals.

---

## Step 10 · Your own index ([`index.py`](index.py))

### Exact search
```python
sims = queries @ self.vectors.T
top = np.argpartition(-sims, k, axis=1)[:, :k]     # O(n): no full sort needed
```
One matrix product gives every similarity; `argpartition` finds the top k without sorting all 200,000. This is the ground truth for Step 11.

### A tiny HNSW
HNSW (hierarchical navigable small world) is a layered graph:
- Every point is in layer 0. Each point also gets a random top layer, `level = int(-log(1 - u) * ml)` with `ml = 1/log(m)`, so each layer holds about 1/m of the one below: 1,500 / 249 / 33 / 9 / 1 points here.
- **Insert:** walk greedily down from the top to the new point's level, then on each lower layer run a small search and link to the m closest; neighbours keep only their best links (2m on layer 0, which carries the real search).
- **Search** (`search`): on each sparse upper layer, `_greedy` moves to whichever neighbour is closer to the query until none is. Upper layers are highways: few points, long jumps. On layer 0, `_search_layer` keeps the `ef` best candidates found so far, so it can step around a local dead end. Larger `ef` = more candidates = higher recall, more time.

On 1,500 2D points it reaches recall@5 of 0.998 with `ef` 5 and 1.0 from `ef` 10. [`hnsw_gif.py`](hnsw_gif.py) records every step through `trace` and draws the GIF: the query enters at the top and descends layer by layer.

---

## Step 11 · Approximate search at scale ([`embed_wiki.py`](embed_wiki.py), [`ann_benchmark.py`](ann_benchmark.py))

**Why another corpus:** the insurance corpus has about 1,100 chunks, about 180 times fewer than the benchmark below, so exact search over it is trivial and no index is needed. To see when an index pays off, `embed_wiki.py` embeds 200,000 Slovak Wikipedia passages from project 12 (reservoir sampling, so the 300 MB file never has to fit in memory) plus 2,000 held-out queries, in resumable shards.

```python
index = hnswlib.Index(space="ip", dim=1024); index.init_index(max_elements=n, M=m, ef_construction=200)
index = faiss.IndexHNSWFlat(1024, 16, faiss.METRIC_INNER_PRODUCT)
index = faiss.IndexIVFFlat(quantizer, 1024, 1024, faiss.METRIC_INNER_PRODUCT)   # 1,024 clusters
```

**How it is measured:**
- **Recall@10** against `BruteForce`'s exact answer, not against relevance: this measures the index, not the model.
- **Latency** one query at a time on one thread (`threadpool_limits(1)`), median and p95 over 500 queries, because that is how a request meets an index.
- **Build time and size**, from the saved index.
- Sweeps: HNSW `M` 8/16/32 × `ef` 10–400; IVF `nprobe` 1–128 (how many of the 1,024 clusters to scan).

**What came out:** exact search 5.45 ms; FAISS HNSW M16 ef50 0.19 ms at recall 0.955; IVF fastest to build (2.8 s). Size barely changes because the vectors themselves dominate. Going from recall 0.955 to 0.991 costs about 3× more time: the last few percent are the expensive ones.

---

## Try it yourself

1. Add bge-m3's sparse output or BM25 to the dense ranking with Reciprocal Rank Fusion (`1/(60 + rank)` summed over both lists). Does hybrid search beat dense alone on short questions?
2. Generate training questions with a length limit (under 100 characters) and a val set from a third writer. Does val now pick the better model, and does the short-question loss disappear?
3. Run `TinyHNSW` with `m` 2 and 16 on the 2D points. How do the layer sizes and the recall at `ef` 5 change?

## Recap

| Step | Idea |
|---|---|
| 0 | Quality = model + chunks; speed = index, a separate question |
| 1 | Real PDFs, cleaned, structure kept; offsets frozen after labelling |
| 2 | Split by insurer before any question; measure the leftover overlap |
| 3 | Labels as evidence spans; quote must be found; different writer for test |
| 4 | Golden set read in full, every drop and rewrite in code |
| 5 | One scorer for everyone; 80 % coverage; bootstrap intervals |
| 6 | BM25 0.191, bge-m3 0.574; chunking inside the noise |
| 7 | In-batch negatives = cross-entropy over a similarity matrix → 0.674 |
| 8 | Hard negatives must look like positives, or the model learns the difference |
| 9 | Val from the training writer chose wrongly; never re-pick on test |
| 10 | Brute force = one matrix product; HNSW = highways down to a wide bottom search |
| 11 | 200k vectors: 5.45 ms exact vs 0.19 ms HNSW at recall 0.955 |

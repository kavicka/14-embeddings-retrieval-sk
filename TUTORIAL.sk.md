# Tutoriál · 14 · Embeddingy a vyhľadávanie v slovenských poistných podmienkach

🇬🇧 [English version](TUTORIAL.md) · 📄 [README (výsledky)](README.sk.md)

Tento tutoriál postaví projekt v poradí, v akom by si ho písal: zožeň skutočné dokumenty → rozdeľ ich skôr než čokoľvek iné → označ odpovede ako úseky dôkazu → chunking a hodnotenie na jednom mieste → baseliny → fine-tuning s in-batch negatívami → hard negatives → výber na validácii → vlastný index → približný index vo veľkom. Každý krok vysvetľuje, **čo** kód robí, **prečo** je napísaný takto a **ako** funguje. README má všetky čísla.

**Cieľ:** k otázke klienta po slovensky nájsť v skutočných poistných podmienkach článok, ktorý na ňu odpovedá, zmerať, ako dobre to funguje, a zistiť, či sa oplatí doladiť embedding model a použiť približný index.

**Potrebuješ:** projekt 10 (normy, cosine similarity: prečo skalárny súčin jednotkových vektorov radí rovnako ako cosine), projekt 12 (jeho text slovenskej Wikipédie, len na benchmark rýchlosti) a Ollamu na písanie otázok. Knižnice: `sentence-transformers` (bge-m3), `rank_bm25`, `hnswlib`, `faiss-cpu`, `pypdf`.

---

## Krok 0 · Čo je vyhľadávanie, v jednom odseku

Embedding model zmení text na vektor, tu 1 024 čísel, tak, aby texty s podobným významom skončili blízko seba. Vyhľadávanie raz zakóduje každý chunk dokumentov, zakóduje otázku a vráti chunky, ktorých vektory sú najbližšie. Keď majú vektory dĺžku 1, „najbližší“ znamená jednoducho najväčší skalárny súčin (projekt 10). Kvalitu určujú dve veci: **model** (dá „ukradli mi auto“ blízko k „odcudzenie vozidla“?) a **chunky** (je odpoveď vôbec celá v jednom chunku?). Rýchlosť je tretia, samostatná otázka: porovnať otázku s každým vektorom je presné, ale rastie s veľkosťou korpusu; približný index vymení trochu recallu za veľa rýchlosti.

---

## Krok 1 · Skutočné dokumenty, vyčistené, ale so štruktúrou ([`corpus.py`](corpus.py))

```python
SOURCES = [("slsp_majetok", "Slovenská sporiteľňa", "majetok", "https://…pdf"), …]   # 20 verejných PDF
SPLITS = {"test": {"Generali", "UNIQA", "Union"}, "val": {"Wüstenrot", "mBank"}}
```

**Čo:** stiahne 20 verejných všeobecných poistných podmienok (VPP) z 12 zdrojov, vytiahne text cez `pypdf`, vyčistí ho a zapíše `data/docs.jsonl`. PDF a text ostávajú v `data/` a nikdy sa necommitujú; URL sú v kóde, takže korpus si vie ktokoľvek postaviť znova.

**Prečo práve tieto dokumenty:** P7 stavia RAG presne nad takýmto textom, takže poučenia z vyhľadávania sa prenesú priamo. Právnická slovenčina je navyše ťažký prípad: dlhé vety, skloňovanie a formulácie, aké klient nikdy nepoužije.

### Čistenie textu z PDF
```python
text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # spoj slová rozdelené na konci riadka
BROKEN_FONT = str.maketrans({"è": "č", "ò": "ň", "¾": "ľ", …})   # mapa písma, ktorú niektoré PDF stratili
```
- Riadok, ktorý sa opakuje na mnohých stranách, je hlavička alebo päta, a riadok bodiek je obsah; `extract` zahodí oboje.
- Konce odsekov sa **zachovajú**, lebo nesú štruktúru: články, číslované body.
- `quality()` spočíta pre každý dokument diakritiku a bežné slovenské slová. Nezmysly zo stĺpcového rozloženia alebo pokazeného písma sa hneď prejavia skoro nulovým „č, š, ž“ a chýbajúcim „poistenie“.

**Dôležité:** `data/docs.jsonl` sa po označení nesmie zmeniť. Každý label je posun v znakoch do tohto textu (Krok 3), takže nové spustenie čistenia s iným pravidlom by potichu posunulo každú odpoveď.

---

## Krok 2 · Rozdeľ podľa poisťovne, skôr než existuje jediná otázka

```python
def split_of(insurer: str) -> str:
    return next((name for name, group in SPLITS.items() if insurer in group), "train")
```

**Čo:** celé poisťovne idú na jednu stranu: 11 dokumentov train, 3 val (Wüstenrot, mBank), 6 test (Generali, UNIQA, Union).

**Prečo podľa poisťovne, nie podľa dokumentu alebo chunku:** jedna poisťovňa recykluje vlastné formulácie naprieč produktmi. Keby boli podmienky Generali pre autá v train a pre majetok v teste, model by polovicu testového textu už videl. Delenie podľa poisťovne je aj realistický prípad: doladíš na dokumentoch, ktoré máš, a potom príde nová poisťovňa.

### Meranie úniku, ktorý ostane
```python
def leakage(docs):   # podiel 8-slovných úsekov odloženého dokumentu, ktoré sa vyskytujú aj v train
```
Aj medzi poisťovňami sa právne frázy opakujú („poistník je povinný bez zbytočného odkladu…“). 1,6 % až 11,1 % 8-slovných úsekov každého odloženého dokumentu sa nachádza aj v train. Je to dosť málo na to, aby sa test dal nazvať nevideným, a číslo je v README namiesto toho, aby sa skrylo.

---

## Krok 3 · Odpovede ako úseky dôkazu, nie id chunkov ([`make_questions.py`](make_questions.py))

```python
row = {"doc_id": doc["doc_id"], "passage": [start, end], "question": question, "span": span}
```

**Čo:** label je id dokumentu plus posun v znakoch začiatku a konca vety, ktorá odpovedá. O chunkoch nehovorí nič.

**Prečo:** budeš chcieť porovnať veľkosti chunkov (Krok 5). Keby bol label „chunk 412“, každý nový chunking by potreboval nové labely, znova ručne. S úsekom sa dá každý chunking ohodnotiť automaticky: chunk je relevantný, ak úsek pokrýva.

### Ako vznikne otázka
1. `passages()` rozreže každý dokument na súvislé celé odseky s 300 až 900 znakmi: dosť dlhé na odpoveď, dosť krátke, aby sa model sústredil.
2. Každý dokument dostane svoj podiel úryvkov striedavo, takže jedno 60-stranové PDF nedodá väčšinu otázok.
3. Lokálny model dostane úryvok a pokyn: predstav si klienta, ktorý nepozná terminológiu, pýtaj sa vlastnými slovami, opíš situáciu, nepreberaj formulácie; potom **doslova skopíruj** vetu, ktorá odpovedá.
4. `locate()` hľadá citát v úryvku. Ak tam nie je, otázka vypadne: model si odpoveď vymyslel.

### Párovanie, ktoré stratilo 40 % otázok
```python
kept = [(i, c.lower()) for i, c in enumerate(text[start:end]) if c.isalnum()]
needle = "".join(c.lower() for c in quote if c.isalnum())
```
Prvá verzia porovnávala citáty presne a 40 % odmietla. Model neklamal: extrakcia z PDF nechala v zdroji „b ola“ a „pre -\nskúmať“, kým model napísal čisté slovo. Porovnávanie len písmen a číslic a následné premietnutie nálezu späť na pôvodné posuny to opravilo. `second_pass()` sa na stratené úryvky spýtal znova s lepším párovaním.

### Rôzni pisatelia, zámerne
```python
WRITERS = {"golden": "gemma3:12b", "train": "qwen3:30b-a3b…", "val": "qwen3:30b-a3b…"}
```
Golden (testové) otázky píše iný model než tréningové, takže doladený model nemôže vyhrať tým, že spozná štýl jedného pisateľa. Zapamätaj si, že val má rovnakého pisateľa ako train: tento detail rozhodne Krok 9.

---

## Krok 4 · Golden set prečítaný ručne ([`review_golden.py`](review_golden.py))

```python
DROPPED = {"evidence_does_not_answer": [24, 26, …], "fragment_too_short_to_judge": […], …}
REWRITTEN = {6: "Kedy presne začne platiť moje poistenie auta, keď podpíšem zmluvu?", …}
```

**Čo:** všetkých 190 kandidátov do golden setu bolo prečítaných jeden po druhom. 49 vypadlo (16 neprirodzených alebo mimo produktu, 13 s dôkazom, ktorý neodpovedal, 10 útržkov, 8 duplikátov, 2 vyžadujúce otvorený dokument) a 20 bolo preformulovaných. V `golden.json` ich ostalo 141.

**Prečo sú rozhodnutia v kóde:** ktokoľvek vidí, ktorá otázka vypadla a prečo, a nové spustenie skriptu golden set zreprodukuje. Úsek preformulovanej otázky ostáva taký, aký vznikol, takže miesto odpovede je stále overené strojovo.

**Prečo sa celý číta len golden set:** rozhoduje o každom čísle v README. Tréningové páry znesú trochu šumu; z nich sa čítal každý 10. (140 párov, 11 s voľnejšie sediacim dôkazom, ostali).

---

## Krok 5 · Chunking a hodnotenie v jednom súbore ([`common.py`](common.py))

### Tri stratégie chunkingu
```python
if strategy == "fixed200":   spans = fixed_spans(offsets, 200, 40)     # 200 tokenov, prekryv 40
elif strategy == "fixed500": spans = fixed_spans(offsets, 500, 100)
elif strategy == "article":  spans = article_spans(text, offsets)      # rez pri „Článok 5 …“
```
- Veľkosti sa počítajú v **tokenoch vlastného tokenizéra modelu** a `return_offsets_mapping` dá každému tokenu jeho pozíciu v znakoch, takže chunk presne vie, ktoré znaky pokrýva.
- Prekryv znamená, že odpoveď prerezaná hranicou sa celá zmestí do ďalšieho chunku.
- `article_spans` reže pri nadpisoch; článok dlhší ako limit modelu 512 tokenov sa vo vnútri rozdelí na pevné okná, lebo bge-m3 by zvyšok potichu ignoroval.

### Kedy chunk odpovedá?
```python
def relevant(chunk, q, evidence):
    if chunk.doc_id == q["doc_id"]:
        if overlap(chunk, q["span"]) >= COVERAGE * span_length:   # COVERAGE = 0.8
            return True
    return evidence in squeeze(chunk.text)
```
- 80 % pokrytie, nie 100 %: hranica chunku cez posledné slovo odpovede nie je chyba.
- **Alebo tie isté slová kdekoľvek:** jedna poisťovňa môže mať identickú klauzulu v dvoch dokumentoch a nájsť druhú kópiu je správna odpoveď.
- `recall_ceiling` ukáže, na aký podiel otázok vôbec dokáže odpovedať nejaký chunk. Pri 200-tokenových chunkoch je to 98,6 %: pár dlhých odpovedí sa rozdelí na dve ešte pred zoradením.

### Metriky
```python
"recall@5": mean((hits > 0) & (hits <= 5)),      # odpoveď je niekde v top 5
"mrr@10":   mean(found / rank),                   # 1 za 1. miesto, 0,5 za 2. miesto, …
"ndcg@10":  mean(found / log2(rank + 1)),         # miernejšia zľava než MRR
```
`hits` drží pre každú otázku poradie prvého relevantného chunku (0 = nie je v top 10). Pri jednej odpovedi na otázku sa nDCG zúži na zľavu na tom mieste. `with_intervals` robí bootstrap cez otázky: 141 otázok sa 2 000-krát prevzorkuje a vezme sa 2,5. a 97,5. percentil. Pri 141 otázkach sú tie intervaly široké a v Kroku 9 na tom záleží.

**Prečo je všetko v jednom súbore:** BM25, základný model aj oba doladené modely hodnotí to isté `first_hits` a `scores`. Žiadny systém nemôže vyhrať vďaka miernejšiemu hodnoteniu.

---

## Krok 6 · Baseliny: BM25 a bge-m3 ([`evaluate.py`](evaluate.py))

```python
def bm25_rankings(chunks, questions):
    index = BM25Okapi([words(c.text) for c in chunks])
def dense_rankings(model, chunks, questions):
    return BruteForce(encode(model, chunk_texts)).search(encode(model, questions), TOP)
```

**BM25** boduje zhodu slov, vážených tak, aby zriedkavé slová platili viac. Dosiahne recall@5 **0,191**. Klient napíše „ukradli“, kde podmienky hovoria „odcudzenie“, a slovenské skloňovanie spraví z jedného slova veľa tvarov; bez stemmingu BM25 zhodu nevidí.

**bge-m3**, multilingválny embedding model, dosiahne **0,574** bez akéhokoľvek tréningu. `normalize_embeddings=True` dá každému vektoru dĺžku 1, takže skalárny súčin v `BruteForce` je cosine similarity.

**Chunking** bge-m3 takmer nepohne (0,574 / 0,546 / 0,567, v rámci šumu), kým BM25 sa s dlhšími chunkami zlepší: viac slov, viac šancí na zhodu.

---

## Krok 7 · Fine-tuning s in-batch negatívami ([`train_embed.py`](train_embed.py))

```python
columns = [[p["question"] for p in batch], [p["positive"] for p in batch]]
loss = losses.MultipleNegativesRankingLoss(model, scale=20.0)(features, None)
```

**Ako funguje loss:** vezmi batch 8 párov (otázka, úryvok). Zakóduj všetkých 8 otázok a všetkých 8 úryvkov a spočítaj maticu podobností 8×8. Riadok *i* má byť najvyšší v stĺpci *i*, pri svojom úryvku; ostatných 7 úryvkov v batchi sú negatívy zadarmo. Loss je cross-entropy cez každý riadok (projekt 08), s podobnosťami vynásobenými `scale` 20, aby bol softmax dosť ostrý na učenie. Toto je contrastive learning: pritiahni zodpovedajúce páry, odtlač zvyšok.

**Prečo záleží na veľkosti batchu a prečo je 8:** väčší batch znamená viac negatívov na otázku, zvyčajne lepšie. Batch 16 vyčerpal pamäť GPU a macOS proces zabil **bez chyby v Pythone**: tichá smrť môže byť pamäť. Gradient checkpointing (`gradient_checkpointing_enable()`) prepočíta aktivácie pri spätnom prechode namiesto ich ukladania, vymení čas za pamäť.

**Ostatné hyperparametre:** learning rate 2e-5 (malý: model je už dobrý a má sa posunúť, nie zabudnúť), warmup 20 krokov, potom lineárny pokles, 3 epochy, gradienty orezané na 1,0, seed 14, všetko uložené v `results_train_*.json`.

**Výber epochy:** po každej epoche `val_score` spustí vyhľadávanie na val dokumentoch a model sa uloží, len ak sa val MRR@10 zlepšil. Na testové dokumenty sa tu nesiaha.

Výsledok na golden sete: recall@5 0,574 → **0,674**, MRR@10 0,392 → 0,456.

---

## Krok 8 · Hard negatives a chyba, ktorá stojí za zachovanie

```python
row["negative"] = next(pool[j].text for j in ranked[:10]
                       if not overlaps_positive(pool[j]) and evidence not in squeeze(pool[j].text))
columns.append([p["negative"] for p in batch])
```

**Čo:** náhodné in-batch negatívy sú často ľahké (cestovná klauzula vs otázka o aute). Hard negative je úryvok, ktorý **základný model** k otázke chybne zaradí vysoko, takže tréning sa sústredí na zámeny, ktoré sa naozaj dejú. `mine_hard_negatives` vezme top 10 pre každú tréningovú otázku a vyberie prvý, ktorý nie je pozitív ani prekrývajúci sa sused a neobsahuje dôkaz.

**Chyba:** prvá verzia ťažila 200-tokenové chunky, kým pozitívy boli úryvky s 300 až 900 znakmi, a nevynechávala susedov prekrývajúcich odpoveď. Val MRR klesol z 0,433 na 0,19 až 0,28 a recall bol najhorší na dlhých chunkoch. Model sa najskôr naučil „dlhší text je ten zlý“. Negatívy naučia to, čím sa líšia od pozitívov; ak je to dĺžka, model sa naučí dĺžku. Ťaženie v rovnakom formáte úryvkov a vynechanie prekryvov to opravilo. Nepodarený beh ostal v `results_train_hardneg_chunks.json`.

---

## Krok 9 · Vyber na val, reportuj na teste ([`select_model.py`](select_model.py))

```python
chosen = max(best, key=lambda n: best[n]["val_mrr@10"])
```

**Pravidlo, zafixované skôr, než existovali golden čísla:** model s najvyšším val MRR@10. Hard negatives vyhrali na val (0,610 vs 0,569).

**Čo povedal test:** in-batch recall@5 0,674, hard negatives 0,617. Val vybrala slabší model a vybraný model nemá preukázateľné zlepšenie oproti základnému (+0,043, interval −0,071 až 0,142).

**Prečo sa výber nezmenil:** výber po pohľade na test spraví z testu druhú validačnú sadu a reportované číslo prestane niečo znamenať. Oba riadky ostávajú v tabuľke aj s vysvetlením.

**Prečo val zlyhala** ([`analysis.py`](analysis.py)): ukáže to rozdelenie golden otázok podľa dĺžky. Pod 80 znakov základný 0,634 vs doladené 0,537 / 0,488; 80 znakov a viac základný 0,550 vs doladené 0,730 / 0,670. Tréningové otázky mali priemerne 144 znakov, golden 108. Val pochádzala od rovnakého pisateľa ako train (141 znakov), takže odmenila presne ten sklon, ktorý na teste uškodil. **Dáta na výber sa majú od tréningových líšiť tak, ako sa líši produkcia.**

`analysis.py` robí aj bootstrap **párových** rozdielov: tá istá otázka na oboch stranách, takže obtiažnosť jednotlivých otázok sa vyruší a interval je užší, než keby sa porovnávali dva samostatné intervaly.

---

## Krok 10 · Vlastný index ([`index.py`](index.py))

### Presné hľadanie
```python
sims = queries @ self.vectors.T
top = np.argpartition(-sims, k, axis=1)[:, :k]     # O(n): netreba triediť všetko
```
Jeden maticový súčin dá všetky podobnosti; `argpartition` nájde top k bez triedenia všetkých 200 000. Toto je pravda, voči ktorej sa meria Krok 11.

### Malé HNSW
HNSW (hierarchical navigable small world) je vrstvený graf:
- Každý bod je vo vrstve 0. Každý bod dostane aj náhodnú najvyššiu vrstvu, `level = int(-log(1 - u) * ml)` s `ml = 1/log(m)`, takže každá vrstva má asi 1/m bodov vrstvy pod ňou: tu 1 500 / 249 / 33 / 9 / 1.
- **Vkladanie:** chamtivo zostúp zhora na úroveň nového bodu, potom na každej nižšej vrstve sprav malé hľadanie a prepoj ho s m najbližšími; susedia si nechajú len svoje najlepšie spojenia (2m vo vrstve 0, ktorá nesie skutočné hľadanie).
- **Hľadanie** (`search`): na každej riedkej hornej vrstve sa `_greedy` presúva na suseda bližšieho k otázke, kým taký existuje. Horné vrstvy sú diaľnice: málo bodov, dlhé skoky. Vo vrstve 0 drží `_search_layer` `ef` najlepších doteraz nájdených kandidátov, takže vie obísť lokálnu slepú uličku. Väčšie `ef` = viac kandidátov = vyšší recall, viac času.

Na 1 500 bodoch v 2D dosiahne recall@5 0,998 pri `ef` 5 a 1,0 od `ef` 10. [`hnsw_gif.py`](hnsw_gif.py) zaznamená cez `trace` každý krok a nakreslí GIF: otázka vstúpi hore a zostupuje vrstvu po vrstve.

---

## Krok 11 · Približné hľadanie vo veľkom ([`embed_wiki.py`](embed_wiki.py), [`ann_benchmark.py`](ann_benchmark.py))

**Prečo iný korpus:** poistný korpus má asi 1 100 chunkov, asi 180-krát menej než benchmark nižšie, takže presné hľadanie v ňom je triviálne a index netreba. Aby bolo vidieť, kedy sa index oplatí, `embed_wiki.py` zakóduje 200 000 úryvkov slovenskej Wikipédie z projektu 12 (reservoir sampling, takže 300 MB súbor sa nikdy nemusí zmestiť do pamäte) plus 2 000 odložených otázok, po častiach, ktoré sa dajú obnoviť.

```python
index = hnswlib.Index(space="ip", dim=1024); index.init_index(max_elements=n, M=m, ef_construction=200)
index = faiss.IndexHNSWFlat(1024, 16, faiss.METRIC_INNER_PRODUCT)
index = faiss.IndexIVFFlat(quantizer, 1024, 1024, faiss.METRIC_INNER_PRODUCT)   # 1 024 zhlukov
```

**Ako sa meria:**
- **Recall@10** voči presnej odpovedi `BruteForce`, nie voči relevancii: meria sa index, nie model.
- **Latencia** po jednej otázke na jednom vlákne (`threadpool_limits(1)`), medián a p95 cez 500 otázok, lebo tak sa požiadavka stretne s indexom.
- **Čas stavby a veľkosť** z uloženého indexu.
- Sweep: HNSW `M` 8/16/32 × `ef` 10 až 400; IVF `nprobe` 1 až 128 (koľko z 1 024 zhlukov sa prehľadá).

**Čo vyšlo:** presné hľadanie 5,45 ms; FAISS HNSW M16 ef50 0,19 ms pri recall 0,955; IVF sa postaví najrýchlejšie (2,8 s). Veľkosť sa takmer nemení, lebo dominujú samotné vektory. Prechod z recall 0,955 na 0,991 stojí asi 3× viac času: posledných pár percent je najdrahších.

---

## Vyskúšaj si sám

1. Pridaj k hustému zoradeniu sparse výstup bge-m3 alebo BM25 cez Reciprocal Rank Fusion (`1/(60 + rank)` sčítané cez oba zoznamy). Porazí hybridné hľadanie samotné husté na krátkych otázkach?
2. Vygeneruj tréningové otázky s limitom dĺžky (pod 100 znakov) a val od tretieho pisateľa. Vyberie teraz val lepší model a zmizne strata na krátkych otázkach?
3. Spusti `TinyHNSW` s `m` 2 a 16 na 2D bodoch. Ako sa zmenia veľkosti vrstiev a recall pri `ef` 5?

## Zhrnutie

| Krok | Myšlienka |
|---|---|
| 0 | Kvalita = model + chunky; rýchlosť = index, samostatná otázka |
| 1 | Skutočné PDF, vyčistené, so štruktúrou; posuny po označení zamrznuté |
| 2 | Rozdelenie podľa poisťovne pred prvou otázkou; zmeraj zvyšný prekryv |
| 3 | Labely ako úseky dôkazu; citát sa musí nájsť; iný pisateľ pre test |
| 4 | Golden set prečítaný celý, každé vyradenie a prepis v kóde |
| 5 | Jedno hodnotenie pre všetkých; 80 % pokrytie; bootstrap intervaly |
| 6 | BM25 0,191, bge-m3 0,574; chunking v rámci šumu |
| 7 | In-batch negatívy = cross-entropy cez maticu podobností → 0,674 |
| 8 | Hard negatives musia vyzerať ako pozitívy, inak sa model naučí rozdiel |
| 9 | Val od tréningového pisateľa vybrala zle; nikdy nevyberaj znova na teste |
| 10 | Brute force = jeden maticový súčin; HNSW = diaľnice dole k širokému hľadaniu |
| 11 | 200k vektorov: 5,45 ms presne vs 0,19 ms HNSW pri recall 0,955 |

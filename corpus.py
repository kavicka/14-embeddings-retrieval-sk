"""Public Slovak insurance terms: download the PDFs, extract text, keep the article structure.

The PDFs stay in data/ and are never committed. Offsets into data/docs.jsonl are the coordinates of every label, so
this file must not change after labelling: any chunking can then be scored against the same evidence spans.
"""

import json
import re
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

# id, insurer, product, url
SOURCES = [
    ("slsp_majetok", "Slovenská sporiteľňa", "majetok", "https://cdn0.erstegroup.com/content/dam/sk/slsp/www_slsp_sk/documents/Ine/vseobecne-poistne-podmienky-pre-poistenie-majetku-a-zodpovednosti-za-skodu-obcanov-pre-klientov-slovenskej-sporitelne-01082020.pdf"),
    ("allianz_domov", "Allianz", "majetok", "https://www.allianz.sk/content/dam/onemarketing/cee/azsk/majetok/môj-domov/Všeobecné-poistné-podmienky-(schválené-dňa-30.9.2016).pdf"),
    ("union_vpp", "Union", "všeobecné", "https://www.union.sk/wp-content/uploads/2022/11/Vseobecne-poistne-podmienky-platne-od-1.1.2021.pdf"),
    ("csob_domos", "ČSOB", "majetok", "https://csob.sk/documents/11005/99131/VPP_pre_Domos_Kompakt_od_18102024.pdf"),
    ("colonnade_cestovne", "Colonnade", "cestovné", "https://citi.com/icg/sa/emea/slovakia/assets/docs/Cestovne_a_urazove_poistenie_colonnade_poistne_podmienky_platne_od_01.02.2020.pdf"),
    ("kp_tuzemske", "Komunálna", "cestovné", "https://www.kpas.sk/content/files/PP/NezivotnePoistenie/202301/Všeobecné poistné podmienky pre tuzemské cestovné poistenie (VPP 393-5).pdf"),
    ("kp_eurotravel", "Komunálna", "cestovné", "https://www.kpas.sk/content/files/PP/NezivotnePoistenie/202301/Všeobecné poistné podmienky pre cestovné poistenie v zahraničí  EUROTRAVEL - ONLINE  (VPP 396-3).pdf"),
    ("kp_pzp", "Komunálna", "motor", "https://www.kpas.sk/content/files/Všeobecné poistné podmienky pre poistenie zodpovednosti za škodu spôsobenú prevádzkou motorového vozidla (VPP 705-11).pdf"),
    ("uniqa_cestovne", "UNIQA", "cestovné", "https://www.uniqa.sk/documents/uniqa_dokumenty/produkty/cestovne-poistenie-na-slovensku/uniqa-do-31.8.2021/cp-164-vpp-cp-sk-18.pdf"),
    ("uniqa_dcr", "UNIQA", "cestovné", "https://www.uniqa.sk/documents/uniqa_dokumenty/produkty/cestovne-poistenie-na-slovensku/uniqa-do-31.8.2021/cp-177-vpp-dcr-h-19-pr-2.pdf"),
    ("wustenrot_rpvs", "Wüstenrot", "zodpovednosť", "https://wuestenrot.sk/wp-content/uploads/2024/10/VPP_RPVS_od_07_10_2024.pdf"),
    ("kooperativa_tuzemske", "Kooperativa", "cestovné", "https://universal.sk/wp-content/uploads/documents_old-web/VPP_Tuzemské%20CP1606811078009.pdf"),
    ("zodpovednost_zamestnanca", "neuvedená", "zodpovednosť", "https://universal.sk/wp-content/uploads/documents_old-web/VPP%20PRE%20POISTENIE%20ZODPOVEDNOSTI%20ZA%20ŠKODU%20SPÔSOBENÚ%20PRI%20VÝKONE%20POVOLANIA%20ZAM%2020191606833403049.pdf"),
    ("mbank_karty", "mBank", "cestovné", "https://www.mbank.sk/pdf-web/poistenie/VPP_OPP_Cestovne_karty.pdf"),
    ("generali_hav", "Generali", "motor", "https://www.generali.sk/wp-content/uploads/2024/05/VPP_HAV_16.pdf"),
    ("generali_hav_poruchy", "Generali", "motor", "https://www.generali.sk/wp-content/uploads/2019/02/vpp-hav-17_2019_fin.pdf"),
    ("generali_majetok", "Generali", "majetok", "https://www.generali.sk/wp-content/uploads/2023/08/VPPMaZFO2017_v3_04_2023.pdf"),
    ("kooperativa_hav", "Kooperativa", "motor", "https://kooperativa.sk/content/userfiles/insuranceterms/1533282754852.pdf"),
    ("kp_prodomo", "Komunálna", "majetok", "https://www.kpas.sk/content/files/vseobecne-poistne-podmienky-pre-poistenie-majetku-a-zodpovednosti-za-skodu-obcanov-prodomo-vpp-mzo-5-p.pdf"),
    ("wustenrot_domov", "Wüstenrot", "majetok", "https://www.wuestenrot.sk/wp-content/uploads/2026/01/VPP_AS_W_dobrom_domov_od_10_05_2025.pdf"),
]  # fmt: skip
# whole insurers go to one side: their documents share wording, and a split inside an insurer would leak it
SPLITS = {"test": {"Generali", "UNIQA", "Union"}, "val": {"Wüstenrot", "mBank"}}
PDF_DIR = Path("data/pdf")
HEADING = re.compile(r"^(?:Článok|ČLÁNOK|Čl\.)\s*\d+[^\n]{0,90}$|^Článok\s+[IVXLC]+[^\n]{0,90}$", re.MULTILINE)


def fetch(doc_id: str, url: str) -> Path | None:
    path = PDF_DIR / f"{doc_id}.pdf"
    if path.exists():
        return path
    safe = urllib.parse.quote(url, safe=":/%()&=?.-_~+,")
    req = urllib.request.Request(safe, headers={"User-Agent": "Mozilla/5.0 (research; learning project)"})
    try:
        data = urllib.request.urlopen(req, timeout=60).read()
    except OSError as err:  # a dead link must not stop the other documents
        print("FAILED", doc_id, repr(err)[:90], flush=True)
        return None
    if not data.startswith(b"%PDF"):
        print("NOT A PDF", doc_id, flush=True)
        return None
    path.write_bytes(data)
    return path


# a font map that some PDFs lost: these Latin-1 letters stand for Central European ones
BROKEN_FONT = str.maketrans({"è": "č", "ò": "ň", "¾": "ľ", "ï": "ď", "»": "ť", "¼": "Ľ", "È": "Č", "Ò": "Ň", "Ï": "Ď"})


def clean(text: str) -> str:
    """Join words hyphenated across lines and page furniture, but keep paragraph breaks: they carry the structure."""
    text = unicodedata.normalize("NFC", text).replace("­", "").replace("\xa0", " ")
    if text.count("è") + text.count("ò") > 3:
        text = text.translate(BROKEN_FONT)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("\uf0b7", "•").replace("\uf020", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract(path: Path) -> tuple[str, int]:
    reader = PdfReader(path)
    pages = [clean(p.extract_text() or "") for p in reader.pages]
    # a line repeated across many pages is a header or footer, and a dotted line is the table of contents
    seen = Counter(re.sub(r"\d+", "#", ln.strip()) for p in pages for ln in set(p.splitlines()) if ln.strip())
    limit = max(3, len(pages) // 3)
    keep = []
    for page in pages:
        lines = [
            ln
            for ln in page.splitlines()
            if seen[re.sub(r"\d+", "#", ln.strip())] < limit and not re.search(r"\.{5,}|…{2,}|\ufffd{3,}", ln)
        ]
        keep.append("\n".join(lines))
    return "\n\n".join(p for p in keep if p.strip()), len(pages)


def quality(text: str) -> dict:
    """Column layouts and old fonts give garbage. Slovak diacritics and common words reveal whether the text is real."""
    letters = sum(c.isalpha() for c in text) or 1
    return {
        "chars": len(text),
        "diacritics_pct": round(100 * sum(c in "áäčďéíĺľňóôŕšťúýž" for c in text.lower()) / letters, 2),
        "stopwords_per_1k": round(
            1000
            * len(re.findall(r"\b(?:poistenia|poistenie|poistnej|alebo|ktorý|ktorá|ktoré)\b", text.lower()))
            / max(len(text), 1),
            2,
        ),
        "short_line_pct": round(
            100 * sum(len(ln) < 20 for ln in text.splitlines()) / max(len(text.splitlines()), 1), 1
        ),
        "headings": len(HEADING.findall(text)),
    }


def split_of(insurer: str) -> str:
    return next((name for name, group in SPLITS.items() if insurer in group), "train")


def shingles(text: str, n: int = 8) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def leakage(docs: list[dict]) -> dict:
    """Share of each held-out document's 8-word runs that also occur in a training document: shared boilerplate."""
    train = set().union(*(shingles(d["text"]) for d in docs if d["split"] == "train"))
    out = {}
    for d in docs:
        if d["split"] != "train":
            own = shingles(d["text"])
            out[d["doc_id"]] = round(100 * len(own & train) / max(len(own), 1), 1)
    return out


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    for doc_id, insurer, product, url in SOURCES:
        path = fetch(doc_id, url)
        if path is None:
            continue
        text, pages = extract(path)
        q = quality(text)
        docs.append(
            {
                "doc_id": doc_id,
                "insurer": insurer,
                "product": product,
                "split": split_of(insurer),
                "url": url,
                "pages": pages,
                "text": text,
                **q,
            }
        )
        print(f"{doc_id:26} {insurer:20} {product:12} p{pages:3} {q}", flush=True)
    with open("data/docs.jsonl", "w") as out:
        out.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in docs)
    meta = [
        {k: d[k] for k in ("doc_id", "insurer", "product", "split", "url", "pages", "chars", "headings")} for d in docs
    ]
    summary = {
        "documents": meta,
        "failed": [s[0] for s in SOURCES if s[0] not in {d["doc_id"] for d in docs}],
        "overlap_with_train_pct": leakage(docs),
    }
    json.dump(summary, open("results_corpus.json", "w"), indent=1, ensure_ascii=False)
    print(summary["overlap_with_train_pct"], summary["failed"])


if __name__ == "__main__":
    main()

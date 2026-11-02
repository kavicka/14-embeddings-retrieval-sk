"""The golden set after reading all 190 candidates one by one. Claude did this reading, not the owner.

Numbers are positions in data/golden_candidates.json (sorted by document and offset). Every drop has a reason, every
rewrite is listed, and the evidence span of a rewritten question stays as generated.
"""

import json
from collections import Counter

DROPPED = {
    "evidence_does_not_answer": [24, 26, 55, 66, 73, 76, 77, 97, 116, 142, 168, 182, 187],
    "fragment_too_short_to_judge": [2, 20, 37, 51, 52, 62, 87, 88, 91, 124],
    "duplicate_of_another_question": [14, 33, 41, 43, 46, 47, 136, 159],
    "needs_the_document_open": [15, 58],
    "unnatural_or_off_product": [25, 29, 48, 65, 68, 69, 70, 78, 79, 80, 82, 95, 114, 117, 132, 141],
}
REWRITTEN = {
    6: "Kedy presne začne platiť moje poistenie auta, keď podpíšem zmluvu?",
    7: "Ak zomrie môj manžel a auto sme vlastnili spolu, kto je po ňom poistený?",
    10: "Ak mi po poistnej udalosti ostanú z auta použiteľné diely, komu patria?",
    21: "Ak môj blízky zomrie na následky nehody do troch rokov od nej, vyplatí poisťovňa peniaze?",
    49: "Mám v aute systém na vyhľadanie po krádeži. Na akom území musí fungovať?",
    54: "Ak sa mi ukradnutá batožina po čase nájde, musím to poisťovni oznámiť?",
    63: "Ak sa mi list od poisťovne vráti ako nedoručiteľný, lebo som sa odsťahoval, kedy sa považuje za doručený?",
    72: "Ak vo vnútri kotla prebehne chemická reakcia a poškodí ho, hoci steny nepraskli, berie sa to ako výbuch?",
    85: "Mám doma alarm so sirénou. Zvýši sa mi tým limit plnenia pri vlámaní?",
    86: "Aká hrubá musí byť ochranná fólia na skle, aby ju poisťovňa uznala?",
    90: "Môžem mať na presklených balkónových dverách zvnútra zasunutý kľúč?",
    99: "Ak sa s manželom rozídeme a majetok si rozdelíme, kto bude poistníkom?",
    100: "Ak mi do domu vystúpi voda z kanalizácie a nešlo o povodeň, je to poistené?",
    109: "Ak treba po požiari odpratať sutiny z domu, preplatí to poisťovňa?",
    110: "Ak sa po poistnej udalosti musím dočasne vysťahovať do inej budovy, platí poistenie aj tam?",
    135: "Ak chcem na dovolenke letieť na paraglide, vzťahuje sa na to poistenie?",
    144: "Ak ma niekto zavraždí do troch rokov od uzavretia zmluvy, dostanú pozostalí plnenie?",
    152: "Ak mi batožinu ukradne môj brat, bude poisťovňa platiť?",
    179: "Na aké číslo mám zavolať, ak sa v horách stratím a potrebujem pomoc?",
    185: "Čo ak mi veci ukradnú bez toho, aby zlodej musel niečo vypáčiť?",
}


def main() -> None:
    candidates = json.load(open("data/golden_candidates.json"))
    reason = {i: why for why, ids in DROPPED.items() for i in ids}
    golden = []
    for n, row in enumerate(candidates):
        if n in reason:
            continue
        golden.append(
            {
                "question": REWRITTEN.get(n, row["question"]),
                "doc_id": row["doc_id"],
                "span": row["span"],
                "rewritten": n in REWRITTEN,
            }
        )
    json.dump(golden, open("golden.json", "w"), indent=1, ensure_ascii=False)
    summary = {
        "generated": 220,
        "extra_tries_for_lost_quotes": 87,
        "candidates_with_quote": len(candidates),
        "dropped": len(reason),
        "dropped_by_reason": {k: len(v) for k, v in DROPPED.items()},
        "rewritten": len(REWRITTEN),
        "golden": len(golden),
        "by_document": Counter(g["doc_id"] for g in golden),
    }
    json.dump(summary, open("results_golden.json", "w"), indent=1, ensure_ascii=False)
    print(summary)


if __name__ == "__main__":
    main()

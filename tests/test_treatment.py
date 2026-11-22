"""Sentence-level citation treatment rules (sentences adapted from the 2025 samples)."""

import pytest

from kg_create import refine_case_treatment, split_sentences, validate_extraction, empty_extraction


def pages_of(*texts):
    return [{"page": i, "text": t, "raw_text": t} for i, t in enumerate(texts, start=1)]


def refine(text, name="Taherakhatoon v. Salambin Mohammad", treatment="CITES", **extra):
    cited = {"name": name, "citation": "", "evidence": "", "treatment": treatment, **extra}
    return refine_case_treatment(cited, pages_of(text))


def test_split_sentences_keeps_legal_abbreviations_together():
    text = (
        "In Anvar P.V. v. P.K. Basheer & Ors., (2014) 10 SCC 473, the Court held so. "
        "See s. 65B of the Act. No. 5 was filed by Mr. Rao. Then it ended."
    )
    sentences = [text[a:b].strip() for a, b in split_sentences(text)]
    assert sentences == [
        "In Anvar P.V. v. P.K. Basheer & Ors., (2014) 10 SCC 473, the Court held so.",
        "See s. 65B of the Act.",
        "No. 5 was filed by Mr. Rao.",
        "Then it ended.",
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Taherakhatoon (supra) is not on limited notice; hence, it is distinguishable.", "DISTINGUISHES"),
        ("The facts of that case were entirely different from the present one. "
         "Taherakhatoon (supra) does not apply here.", "DISTINGUISHES"),
        ("The view taken in Taherakhatoon (supra) does not lay down the correct law. "
         "It requires to be overruled and we do so.", "OVERRULES"),
        ("Taherakhatoon (supra) is hereby overruled.", "OVERRULES"),
        ("We respectfully agree with the view expressed in Taherakhatoon (supra).", "FOLLOWS"),
        ("The issue is squarely covered by Taherakhatoon (supra).", "FOLLOWS"),
        ("The present case is squarely covered by the decision in Taherakhatoon (supra).", "FOLLOWS"),
        ("Applying the test laid down in Taherakhatoon (supra), the appeal must fail.", "APPLIES"),
        ("We may also rely on Taherakhatoon (supra) where this Court held the same.", "RELIES_ON"),
        ("Reliance can be placed on Taherakhatoon (supra) for this proposition.", "RELIES_ON"),
        # The cue may sit in the sentence right after the mention.
        ("In Taherakhatoon (supra) this Court held that leave is discretionary. "
         "We respectfully agree.", "FOLLOWS"),
    ],
)
def test_judicial_cues(text, expected):
    assert refine(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        # Negated cues.
        "Taherakhatoon (supra) is not distinguishable on facts.",
        "We are conscious that Taherakhatoon (supra) has not been overruled.",
        "Taherakhatoon (supra) cannot be distinguished.",
        # Reporting another court's overruling, not ours.
        "In Anvar v. Basheer, Taherakhatoon (supra) was overruled.",
        # Generic words that the old rules keyed on.
        "Taherakhatoon (supra) said that leave is discretionary.",
        # Cue in a sentence that does not mention the case.
        "Taherakhatoon (supra) concerned a lease. Some other decision is distinguishable. "
        "The third sentence is distinguishable as well.",
    ],
)
def test_no_rule_fires(text):
    assert refine(text) is None


def test_counsel_voice_does_not_upgrade():
    text = (
        "Learned counsel for the appellant placed reliance on Taherakhatoon (supra) "
        "and submitted that it squarely covers the case."
    )
    assert refine(text) is None
    assert refine(text, treatment="RELIES_ON") == "REFERRED_TO"


def test_counsel_voice_downgrade_needs_every_mention_to_be_counsel():
    text = (
        "It was contended that Taherakhatoon (supra) applies. "
        "A neutral sentence follows here. "
        "This Court in Taherakhatoon (supra) considered the same Act. Nothing more."
    )
    assert refine(text, treatment="RELIES_ON") is None


def test_counsel_distinguishing_is_not_the_court_distinguishing():
    text = (
        "It was also submitted that the decision in Taherakhatoon (supra), relied upon by "
        "the High Court, is clearly distinguishable on facts."
    )
    assert refine(text) is None


def test_mentions_found_by_citation_and_evidence():
    text = "The judgment reported in (2015) 5 SCC 223 is clearly distinguishable."
    assert refine(text, name="Rathnavati v. Kavita", citation="(2015) 5 SCC 223") == "DISTINGUISHES"
    assert refine(text, name="Rathnavati v. Kavita", evidence="reported in (2015) 5 SCC 223") == "DISTINGUISHES"


def test_case_not_mentioned_returns_none():
    assert refine("Nothing about the case is distinguishable here.") is None


def test_validate_extraction_records_treatment_source():
    pages = pages_of("We respectfully agree with Taherakhatoon (supra). The appeal fails.")
    extraction = empty_extraction()
    extraction["cited_cases"] = [
        {"name": "Taherakhatoon v. Salambin", "citation": "", "treatment": "REFERRED_TO",
         "proposition": "", "evidence": "We respectfully agree with Taherakhatoon", "page": 1},
        {"name": "Other v. Case", "citation": "", "treatment": "CITES",
         "proposition": "", "evidence": "The appeal fails.", "page": 1},
    ]
    header = {"name": "X v. Y", "court": "", "date": "", "citations": [], "judges": [],
              "author": "", "parties": []}
    record, _ = validate_extraction(extraction, pages, header)
    first, second = record["cited_cases"]
    assert (first["treatment"], first["treatment_source"], first["llm_treatment"]) == (
        "FOLLOWS", "rule", "REFERRED_TO")
    assert (second["treatment"], second["treatment_source"]) == ("CITES", "llm")

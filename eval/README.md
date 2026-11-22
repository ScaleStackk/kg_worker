# Accuracy check

`gold/` is an answer key for the 20 sample judgments listed in
`samples/SELECTED.md` (2025 Supreme Court judgments from Indian Kanoon; the
PDFs are not in this repo).

## How the answer key was made

Nobody on the team is a lawyer, so the key was labelled by a different model
than the pipeline: Claude (Opus 5.5), not Qwen. It read each judgment's
header, signature block, and every sentence containing a citation, and
assigned the labels below. The team spot-checks about 20 items; record that
in "Spot checks" below. Treat the scores as indicative, not authoritative.

- **Header fields (all 20):** court, date, neutral citation, bench (from the
  signature block), author (from the opening "<NAME>, J." line; left null for
  orders with no named author).
- **Cited cases, treatments and core statutes (6 judgments so far):** Hare
  Krushna Mahanta, Urmila Dixit, M/S Ferro Concrete, U. Sudheera, State of
  Rajasthan v. Chatra (5 with citations) plus Chief Revenue Controlling
  Officer (no citations, header only).
  - Included: every case the court itself cites. Excluded: cases cited only
    inside a passage quoted from another judgment.
  - Treatment uses the pipeline's labels. Cases cited only by counsel are
    REFERRED_TO. RELIES_ON means the court adopts the reasoning, APPLIES means
    it applies a stated test or direction, FOLLOWS means it says it follows or
    agrees with the decision, DISTINGUISHES means it explains why the case
    does not govern. The line between RELIES_ON and REFERRED_TO is a
    judgement call, which is why a coarse accuracy is reported too.
  - Statutes: only the statutes the judgment decides under ("core"), not
    every Act named in passing.

## Running it

```bash
# 1. Pipeline output (slow: real Ollama calls). Keep the raw model output.
python kg_create.py ../samples/X.PDF --output out/X.json --debug-output raw/X.raw.json

# 2. Score
python eval/score.py out/

# 3. Re-score another code version on the same model output (fast)
python eval/reprocess.py raw/ ../samples/ out2/ && python eval/score.py out2/
```

Metrics: header-field accuracy; cited-case precision and recall (matched by
citation, else by name); treatment accuracy on matched pairs, strict and
coarse (MENTION = CITES/REFERRED_TO, POSITIVE = RELIES_ON/APPLIES/FOLLOWS,
NEGATIVE = DISTINGUISHES/OVERRULES), with a confusion matrix; statute and
section recall; evidence acceptance rate from `metadata.validation`.

## Spot checks

| Item | Checked by | Agrees? | Note |
|---|---|---|---|
| | | | |

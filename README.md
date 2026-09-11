# Hiver Support Agent — SDE Intern Take-home

A small, auditable support agent for one brand in the Customer Support on Twitter dataset. It classifies a customer tweet, retrieves historical support resolutions, drafts an evidence-bounded reply, and decides whether to auto-handle or escalate.

## Design in one minute

The pipeline selects a brand from customer-to-agent reply pairs, groups records by thread root, and splits **by thread**. The simple baseline is word TF–IDF/logistic regression; the production classifier combines word and character TF–IDF features with logistic regression, making it more resilient to Twitter spelling variation. Retrieval is TF–IDF cosine similarity by default (fast and no model download).

The deliberately conservative policy escalates security, payment, and multi-issue messages; it also escalates low classifier confidence or weak retrieval. Drafts are deterministic, evidence-bounded templates by default. An optional OpenAI generator is available only when `OPENAI_API_KEY` is set; it receives evidence and is instructed not to invent policy.

## Reproduce

Python 3.10+ is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\make_fixture.py
python scripts\run_pipeline.py --input data\fixture_tweets.csv --fixture
pytest -q
```

This verifies the full pipeline in seconds. Fixture outputs are clearly marked `fixture_only` and **must not** be used as assignment results.

For headline results, download `twcs.csv` from Kaggle's `thoughtvector/customer-support-on-twitter` into `data/raw/` (or use `kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw` and unzip it), then run:

```powershell
python scripts\run_pipeline.py --input data\raw\twcs.csv --limit 120000
```

The default cap is intentional: it finishes under 15 minutes on a laptop. It writes all measured artefacts under `results/`. No table in this repository contains invented Kaggle metrics.

## Manual golden-set workflow (required before submission)

The code prepares a stratified 200-row candidate file at `data/golden_candidates.csv`. A human must fill `gold_intent`, `human_auto_handle`, and the five 0–2 reply rubric columns, then run:

```powershell
python scripts\evaluate_golden.py --gold data\golden_annotated.csv
```

`data/golden_annotated.example.csv` documents the schema. Do not call an automatically inferred label “hand-labelled.” The annotation guide is in `data/ANNOTATION_GUIDE.md`.

To create a review-ready worksheet with assistant suggestions kept separate from human labels:

```powershell
python scripts\create_review_sheet.py
```

After human scoring a 50-row calibration subset, set `GEMINI_API_KEY` in a local `.env` file and optionally score the *same* rows with the Gemini judge, then measure agreement:

```powershell
python scripts\llm_judge.py --input data\golden_annotated.csv --output data\golden_with_judge.csv --limit 50
python scripts\evaluate_golden.py --gold data\golden_with_judge.csv
```

## CLI demo

```powershell
python scripts\answer.py --artifacts artifacts --message "I was charged twice for my order"
```

It returns JSON with `intent`, `confidence`, `decision`, `reason`, `evidence`, and `draft_reply`.

## Evaluation

`run_pipeline.py` reports majority and TF–IDF+LR baselines, accuracy, macro/per-intent F1, a confusion matrix, retrieval-isolation checks, and escalation coverage against policy labels. `evaluate_golden.py` reports reply rubric aggregates and judge-vs-human agreement (Spearman, MAE, exact and within-one agreement). The optional judge is a Gemini structured JSON call; human scores always remain the reference.

## Limits

This is a decision-support prototype, not a Twitter integration, a policy engine, autonomous account tooling, or a production safety system. It cannot resolve messages whose resolution depends on private account data.

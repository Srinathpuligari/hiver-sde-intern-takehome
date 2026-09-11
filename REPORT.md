# Report — Hiver Support Agent

## Executive summary

This repository implements an auditable prototype for a single automatically selected Twitter-support brand. Given a customer tweet, it outputs an intent, confidence, historical support evidence, a conservative automation decision, and an evidence-bounded draft. The primary objective is trustworthiness rather than a broad autonomous agent.

**Measured bounded-subsample diagnostic (TWCS, first 120,000 tweets):** the selection rule chose **AmazonHelp** (8,471 direct customer→agent pairs). A thread-level split produced 6,818 train and 1,653 test cases with zero thread overlap. The majority baseline achieved 48.3% accuracy / 5.9% macro F1; TF–IDF+logistic regression achieved 72.9% accuracy / 66.7% macro F1. These are *weak-label consistency checks*, not the final headline result.

**Human-labelled golden evaluation:** 195 held-out examples were reviewed. The production hybrid word-plus-character classifier achieved **57.4% accuracy and 53.3% macro F1**, compared with 53.3% / 51.4% for a word-only classifier. The 50 reply-scored items averaged **6.90/10** across correctness, groundedness, completeness, safety, and tone. This is a deliberately candid result: the current escalation threshold needs calibration before deployment.

## Problem framing

For this brand, “good” means (1) routing a customer message into a useful operational category, (2) avoiding unsupported policy promises, and (3) declining automation when account, payment, security, context, or evidence is inadequate. The implementation does not build Twitter posting, authentication, account/order tools, long-term customer memory, fine-tuning, or an autonomous workflow engine.

The taxonomy contains account access, billing/charge, delivery status, refund request, cancellation, product issue, service outage, information request, complaint, security/privacy, and other. It is a transparent keyword-derived operational proposal; final evaluation labels are human-authored.

## System

The pipeline extracts outbound brand tweets that directly reply to inbound customer tweets. It reconstructs a root identifier from reply links and assigns each whole thread to train or test. The production classifier combines word TF–IDF (1–2 grams) and character TF–IDF (3–5 grams) features with balanced logistic regression. Character features improve resilience to misspellings, handles, and multilingual fragments. Historical retrieval searches only the training cases with cosine similarity. The agent exposes the three most similar cases and either adapts the closest historical reply or escalates. Escalation triggers on high-risk intents, confidence below 0.70, retrieval similarity below 0.20, and possible multi-issue text.

## Evaluation plan

The harness compares two required baselines against the final model: **(1) trivial majority class**, and **(2) simple word TF–IDF plus logistic regression**. It reports accuracy, macro F1, per-intent F1, and a confusion matrix. A 195-item stratified held-out candidate set was human annotated. The reply rubric scores correctness, groundedness, completeness, safety, and tone from 0–2. Gemini scores the same 50 cases and is validated against human labels using Spearman correlation, MAE, exact agreement, and within-one agreement.

Populate this table from generated artefacts only:

| Measure | Result |
|---|---:|
| Majority macro F1 (weak-label diagnostic) | 5.9% |
| TF–IDF+LR macro F1 (weak-label diagnostic) | 66.7% |
| Human-labelled golden examples | 195 |
| Golden majority baseline accuracy / macro F1 | 6.2% / 1.1% |
| Golden-set word-only accuracy / macro F1 | 53.3% / 51.4% |
| Golden-set hybrid accuracy / macro F1 | 57.4% / 53.3% |
| Mean reply quality / 10 (n=50) | 6.90 |
| Gemini-human Spearman / MAE (n=50) | 0.115 / 0.516 |
| Gemini within-one agreement | 98.0% |

## Failure analysis: five observed cases

1. **Login complaint in German.** “...Login/Passwort-Anmeldung...” was labelled `account_access` but predicted `information_request`; cross-lingual wording and profanity obscure the login cue.
2. **Long unresolved account case.** A tweet describing a 54-day investigation was labelled `account_access` but predicted `complaint`; the complaint language dominated the underlying access issue.
3. **Cashback delay.** “...cashback...credited within 7 days...14 days...” was labelled `billing_charge` but predicted `delivery_status`; time-to-arrival vocabulary caused a delivery false match.
4. **Refund destination.** A request to return an order amount to the original payment method was labelled `refund_request` but predicted `billing_charge`; payment-method terminology blurred refund versus billing.
5. **Cancelled undelivered order.** An item marked dispatched, then cancelled, was labelled `cancellation` but predicted `delivery_status`; it contains two valid operational intents but the model is single-label.

All five were escalated, which avoided an unsafe automated reply, but also illustrates low automation coverage. The next iteration should use multi-label routing, add multilingual examples, and train on human labels rather than keyword proposals.

## What is misleading about my headline number?

Even an honest held-out intent score is not an end-to-end trust score. The 72.9% weak-label diagnostic substantially overstates the 57.4% human-golden accuracy, demonstrating that weak labels can inflate automated test results. Class imbalance makes accuracy conceal rare-intent failures; direct reply pairs omit unresolved cases; tweets often need private context; and a 195-item manually labelled set has sampling uncertainty. The Gemini judge is an additional noisy measurement, not ground truth: its 0.115 Spearman correlation with human mean scores is weak despite 98% within-one agreement. Finally, high reply safety scores should not be read as strong automation readiness: this policy escalates most risky cases rather than proving it can resolve them.

## One more week

I would manually label a larger, double-annotated taxonomy dataset; use a sentence embedding retriever with retrieval-relevance labels; add calibrated confidence and selective-risk curves; mine deduplicated failure clusters; and evaluate with brand policy documentation plus a privacy-aware tool handoff.

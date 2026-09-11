# Decision log

1. **Use direct customer→agent reply pairs.** They give the clearest historical problem/resolution evidence.
2. **Select the highest-volume eligible brand automatically.** This makes selection reproducible instead of cherry-picked.
3. **Cap the full input at 120k rows.** It is enough for a credible subsample and fits the 15-minute reproduction constraint.
4. **Split by reconstructed thread root, never tweet.** Replies in a thread can be near duplicates and would leak otherwise.
5. **Index training cases only.** Held-out and golden threads are excluded from retrieval by construction.
6. **Use an 11-intent operational taxonomy.** It trades philosophical completeness for support-routing actions.
7. **Make taxonomy proposals explicit weak labels.** The pipeline never represents them as manual ground truth.
8. **Use majority and TF–IDF+logistic regression baselines.** Both are fast, understandable, and difficult to dismiss as strawmen.
9. **Use cosine TF–IDF retrieval by default.** It is reproducible offline; semantic embeddings are an optional extension.
10. **Return source evidence with every answer.** A reviewer can inspect why a draft was produced.
11. **Use a conservative escalation policy.** Security, payments, multi-issue, weak retrieval, and low confidence should not be automated.
12. **Avoid invented policy language.** The default draft quotes/adapts a retrieved historical reply instead of claiming deadlines or refunds.
13. **Sample the golden set stratified by proposed intent.** Random samples would overrepresent frequent easy classes.
14. **Human scores are the reference for judge calibration.** LLM-judge agreement is reported, not assumed.
15. **Separate fixture from results.** The included test data verifies code but is never presented as Kaggle performance.

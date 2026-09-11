"""Compare reproducible classifiers on the held-out human-labelled golden set."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from src.evaluation import write_json

train = pd.read_csv("artifacts/retrieval_cases.csv")
gold = pd.read_csv("data/golden_annotated.csv")
specs = {
    "word_tfidf_lr": TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=100000),
    "char_tfidf_lr": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True, max_features=150000),
    "hybrid_word_char_tfidf_lr": FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=75000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True, max_features=100000)),
    ]),
}
results = {}
majority = train.proposed_intent.mode().iloc[0]
results["majority_baseline"] = {
    "accuracy": round(float(accuracy_score(gold.gold_intent, [majority] * len(gold))), 3),
    "macro_f1": round(float(f1_score(gold.gold_intent, [majority] * len(gold), average="macro")), 3),
}
for name, vectorizer in specs.items():
    model = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", C=2.0, random_state=42)),
    ])
    predictions = model.fit(train.customer_text, train.proposed_intent).predict(gold.customer_text)
    results[name] = {
        "accuracy": round(float(accuracy_score(gold.gold_intent, predictions)), 3),
        "macro_f1": round(float(f1_score(gold.gold_intent, predictions, average="macro")), 3),
    }
write_json(Path("results") / "model_benchmark.json", results)
for name, metrics in results.items(): print(name, metrics)

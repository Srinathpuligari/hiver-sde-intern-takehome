"""Operational, explainable weak-label taxonomy and classifiers."""
from __future__ import annotations
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from .config import INTENTS

RULES = {
 "security_privacy": r"hack|fraud|security|privacy|stolen|unauthori[sz]ed",
 "account_access": r"login|log in|sign.?in|password|locked|account.*access",
 "billing_charge": r"charged|charge|bill|payment|invoice|debit|credit card",
 "refund_request": r"refund|money back|reimburse",
 "cancellation": r"cancel|unsubscribe|stop my",
 "delivery_status": r"deliver|delivery|shipping|shipment|arriv|tracking|order.*where",
 "service_outage": r"outage|down|not working|error|can.?t (use|access)",
 "product_issue": r"broken|defect|damaged|doesn.?t work|issue with",
 "information_request": r"\bhow\b|\bwhat\b|\bwhere\b|\bwhen\b|\bcan i\b|\?",
 "complaint": r"terrible|worst|disappointed|angry|unacceptable|complaint",
}

def propose_intent(text: str) -> str:
    text = text.lower()
    for label, pattern in RULES.items():
        if re.search(pattern, text): return label
    return "other"

def label_rows(rows):
    rows = rows.copy(); rows["proposed_intent"] = rows.customer_text.map(propose_intent); return rows

def train_classifier(texts, labels):
    # Character n-grams make noisy spelling, handles, and multilingual fragments more robust;
    # word n-grams retain the interpretable support phrases used by the baseline.
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=75000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=100000)),
    ])
    model = Pipeline([("tfidf", features), ("lr", LogisticRegression(max_iter=1500, class_weight="balanced", C=2.0, random_state=42))])
    return model.fit(texts, labels)

def predict(model, texts):
    labels = model.predict(texts); probs = model.predict_proba(texts).max(axis=1)
    return labels, probs

"""Evidence retrieval, bounded drafting, and explicit automation policy."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import HIGH_RISK

@dataclass
class AgentResult:
    intent: str; confidence: float; decision: str; reason: str; evidence: list[dict]; draft_reply: str

class SupportAgent:
    def __init__(self, classifier, retrieval_rows):
        self.classifier = classifier; self.rows = retrieval_rows.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.rows.customer_text)

    def retrieve(self, message, k=3):
        scores = cosine_similarity(self.vectorizer.transform([message]), self.matrix).ravel()
        ids = np.argsort(scores)[::-1][:k]
        return [{"customer":self.rows.iloc[i].customer_text, "historical_reply":self.rows.iloc[i].agent_reply, "similarity":round(float(scores[i]),3), "case_id":str(self.rows.iloc[i].customer_id)} for i in ids]

    @staticmethod
    def _multi_issue(message): return message.lower().count(" and ") >= 1 or ";" in message

    def answer(self, message):
        intent = self.classifier.predict([message])[0]
        confidence = float(self.classifier.predict_proba([message]).max())
        evidence = self.retrieve(message)
        best = evidence[0]["similarity"] if evidence else 0.0
        risks = []
        if intent in HIGH_RISK: risks.append(f"{intent} is high-risk")
        if confidence < .70: risks.append(f"low intent confidence ({confidence:.2f})")
        if best < .20: risks.append(f"weak historical match ({best:.2f})")
        if self._multi_issue(message): risks.append("possible multi-issue message")
        if risks:
            return AgentResult(intent, round(confidence,3), "ESCALATE", "; ".join(risks), evidence, "Thanks for flagging this. I’m routing this to a support specialist so they can review the details securely.")
        reply = evidence[0]["historical_reply"].strip()
        # Preserve historical language rather than create a fresh, ungrounded policy claim.
        draft = f"Thanks for reaching out. Based on similar cases, {reply}"
        return AgentResult(intent, round(confidence,3), "AUTO_HANDLE", "High-confidence intent with a sufficiently similar historical resolution.", evidence, draft)

    def as_dict(self, message): return asdict(self.answer(message))

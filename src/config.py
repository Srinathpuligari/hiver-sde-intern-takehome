from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
RANDOM_STATE = 42
INTENTS = [
    "account_access", "billing_charge", "delivery_status", "refund_request",
    "cancellation", "product_issue", "service_outage", "information_request",
    "complaint", "security_privacy", "other",
]
HIGH_RISK = {"account_access", "billing_charge", "security_privacy"}

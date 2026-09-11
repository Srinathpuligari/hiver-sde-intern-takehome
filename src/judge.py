"""Optional Gemini judge. Never substitutes for the human annotation columns."""
from __future__ import annotations
import json, os
from pathlib import Path
from google import genai
from google.genai import types

RUBRIC = """Score each dimension 0, 1, or 2. Correctness: appropriate support action.
Groundedness: all material claims supported by the evidence. Completeness: addresses the request.
Safety: no invented commitments, unsafe public-data request, or risky handling. Tone: respectful and clear.
Return JSON only with correctness, groundedness, completeness, safety, tone, and rationale."""

def _load_local_env() -> None:
    """Minimal dotenv reader so the scripts work without another dependency."""
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if not env_file.exists(): return
    for line in env_file.read_text(encoding="utf8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

def judge_reply(customer: str, reply: str, evidence_json: str) -> dict:
    _load_local_env()
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required for Gemini judging; use human annotations otherwise.")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
    prompt = f"{RUBRIC}\n\nCustomer: {customer}\nDraft reply: {reply}\nHistorical evidence: {evidence_json}"
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0)
    try:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
    except Exception as error:
        fallback = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
        if fallback == model:
            raise
        response = client.models.generate_content(model=fallback, contents=prompt, config=config)
    result = json.loads(response.text)
    for key in ["correctness","groundedness","completeness","safety","tone"]:
        if result.get(key) not in (0,1,2): raise ValueError(f"Invalid judge score for {key}: {result.get(key)!r}")
    return result

def judge_batch(records: list[dict]) -> list[dict]:
    """Judge several cases per request to stay within free-tier request limits."""
    _load_local_env()
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required for Gemini judging.")
    if not records: return []
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    cases = "\n\n".join(f"CASE {i}:\nCustomer: {r['customer']}\nDraft reply: {r['reply']}\nHistorical evidence: {r['evidence']}" for i, r in enumerate(records))
    prompt = f"{RUBRIC}\n\nScore every case below. Return a JSON object with exactly one key, `scores`, whose value is an array of {len(records)} score objects in the same order.\n\n{cases}"
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0)
    response = client.models.generate_content(model=model, contents=prompt, config=config)
    result = json.loads(response.text).get("scores")
    if not isinstance(result, list) or len(result) != len(records):
        raise ValueError("Gemini returned an unexpected batch score shape.")
    for scores in result:
        for key in ["correctness","groundedness","completeness","safety","tone"]:
            if scores.get(key) not in (0,1,2): raise ValueError(f"Invalid judge score for {key}: {scores.get(key)!r}")
    return result

"""Create a human-review worksheet without misrepresenting model suggestions as labels."""
import argparse
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--input", default="data/golden_with_gemini_scores.csv")
p.add_argument("--output", default="data/golden_review_sheet.csv")
args = p.parse_args()

df = pd.read_csv(args.input)
df["suggested_intent"] = df["agent_intent"]
df["suggested_auto_handle"] = df["decision"].eq("AUTO_HANDLE")
df["assistant_review_note"] = "Review the suggestion against customer_text, draft_reply, and evidence_json. Human fields must be independently entered."

# These remain blank: suggested values are not human labels.
for column in ["gold_intent", "human_auto_handle", "correctness", "groundedness", "completeness", "safety", "tone", "annotation_notes"]:
    df[column] = ""

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.output, index=False)
print(f"Wrote {args.output} with {len(df)} rows and assistant suggestions for review.")

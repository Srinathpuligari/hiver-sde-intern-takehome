"""Add Gemini judge_* columns without overwriting human labels, with checkpoints."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.judge import judge_batch

p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--limit",type=int,default=50); p.add_argument("--batch-size",type=int,default=10); a=p.parse_args()
df=pd.read_csv(a.input)
indices=list(df.index[:a.limit])
for start in range(0,len(indices),a.batch_size):
    batch_indices=indices[start:start+a.batch_size]
    if "judge_correctness" in df:
        pending=[i for i in batch_indices if pd.isna(df.at[i,"judge_correctness"])]
    else:
        pending=batch_indices
    if not pending: continue
    records=[{"customer":str(df.at[i,"customer_text"]),"reply":str(df.at[i,"draft_reply"]),"evidence":str(df.at[i,"evidence_json"])} for i in pending]
    for i,result in zip(pending,judge_batch(records)):
        for key,value in result.items():
            if key != "rationale": df.at[i,"judge_"+key]=value
    df.to_csv(a.output,index=False)
    print(f"Checkpointed {min(start+a.batch_size,len(indices))}/{len(indices)} rows.")
print(f"Wrote {a.output}; Gemini judge scores added for at most {a.limit} rows.")

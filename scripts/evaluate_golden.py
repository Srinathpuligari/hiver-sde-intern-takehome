import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from src.evaluation import agreement, write_json
p=argparse.ArgumentParser();p.add_argument("--gold",required=True);a=p.parse_args(); df=pd.read_csv(a.gold)
required={"gold_intent","human_auto_handle","correctness","groundedness","completeness","safety","tone"}
missing=required-set(df.columns)
if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
scored=df[df.gold_intent.notna() & df.gold_intent.astype(str).ne("")]
summary={"n_human_annotated":len(scored),"reply_rubric_means":{k:round(float(pd.to_numeric(scored[k],errors="coerce").mean()),3) for k in ["correctness","groundedness","completeness","safety","tone"]}}
if len(scored) and "agent_intent" in scored:
    labels=sorted(set(scored.gold_intent) | set(scored.agent_intent))
    report=classification_report(scored.gold_intent, scored.agent_intent, labels=labels, output_dict=True, zero_division=0)
    summary["intent_classification"]={
        "accuracy":round(float(accuracy_score(scored.gold_intent, scored.agent_intent)),3),
        "macro_f1":round(float(report["macro avg"]["f1-score"]),3),
        "per_intent":report,
        "labels":labels,
        "confusion_matrix":confusion_matrix(scored.gold_intent,scored.agent_intent,labels=labels).tolist(),
    }
if len(scored) and {"decision","human_auto_handle"} <= set(scored):
    actual=scored.human_auto_handle.astype(str).str.upper().isin(["TRUE","1","YES"])
    predicted=scored.decision.astype(str).eq("AUTO_HANDLE")
    precision, recall, f1, _=precision_recall_fscore_support(actual,predicted,average="binary",zero_division=0)
    summary["auto_handle"]={"precision":round(float(precision),3),"recall":round(float(recall),3),"f1":round(float(f1),3),"coverage":round(float(predicted.mean()),3),"unsafe_auto_handle_count":int((predicted & ~actual).sum())}
judge_cols=["judge_"+x for x in ["correctness","groundedness","completeness","safety","tone"]]
if all(col in df for col in judge_cols):
    human_cols=["correctness","groundedness","completeness","safety","tone"]
    complete=(df[judge_cols].notna().all(axis=1) & df[human_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)).sum()
    summary["judge_human_agreement"] = agreement(df) if complete >= 3 else {"status":"not_computed", "reason":"Need at least three complete paired human/judge rows.", "paired_rows":int(complete)}
write_json(Path("results")/"golden_evaluation.json",summary); print(summary)

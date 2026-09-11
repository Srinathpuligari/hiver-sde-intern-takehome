from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import spearmanr

def classification_metrics(y_true, y_pred):
    return {"accuracy":round(float(accuracy_score(y_true,y_pred)),4), "per_intent":classification_report(y_true,y_pred,output_dict=True,zero_division=0), "confusion_matrix":confusion_matrix(y_true,y_pred,labels=sorted(set(y_true)|set(y_pred))).tolist(), "labels":sorted(set(y_true)|set(y_pred))}

def write_json(path: Path, value):
    path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(value, indent=2, default=float), encoding="utf8")

def agreement(df: pd.DataFrame):
    rubric=["correctness","groundedness","completeness","safety","tone"]
    human=df[rubric].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    judge=df[["judge_"+x for x in rubric]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    keep=human.notna() & judge.notna()
    if keep.sum()<3: raise ValueError("Need at least three rows with human and judge rubric values.")
    h,j=human[keep],judge[keep]
    return {"n":int(keep.sum()),"spearman":round(float(spearmanr(h,j).statistic),3),"mae":round(float(np.mean(np.abs(h-j))),3),"exact_agreement":round(float(np.mean(h==j)),3),"within_one":round(float(np.mean(np.abs(h-j)<=1)),3)}

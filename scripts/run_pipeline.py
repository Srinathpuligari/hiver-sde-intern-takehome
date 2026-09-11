import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from src.config import ARTIFACTS, RESULTS
from src.data import load_twcs, customer_agent_pairs, choose_brand, split_by_thread, stratified_candidates
from src.intents import label_rows, train_classifier, predict
from src.agent import SupportAgent
from src.evaluation import classification_metrics, write_json

p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--limit",type=int,default=120000); p.add_argument("--fixture",action="store_true"); args=p.parse_args()
df=load_twcs(args.input, None if args.fixture else args.limit)
pairs=customer_agent_pairs(df); brand, stats=choose_brand(pairs, min_pairs=4 if args.fixture else 100)
rows=label_rows(pairs[pairs.brand.eq(brand)].copy())
train,test=split_by_thread(rows,df)
model=train_classifier(train.customer_text,train.proposed_intent)
pred,conf=predict(model,test.customer_text)
majority=train.proposed_intent.mode()[0]
metrics={"data_mode":"fixture_only" if args.fixture else "twcs_subsample", "selected_brand":brand, "brand_stats":stats.head(15).to_dict(orient="index"), "n_train":len(train),"n_test":len(test),"thread_overlap":len(set(train.thread_id)&set(test.thread_id)),"majority_baseline":classification_metrics(test.proposed_intent,[majority]*len(test)),"tfidf_logistic_regression":classification_metrics(test.proposed_intent,pred),"IMPORTANT":"Labels are deterministic taxonomy proposals; replace with human golden labels before reporting headline scores."}
RESULTS.mkdir(exist_ok=True); ARTIFACTS.mkdir(exist_ok=True)
write_json(RESULTS/"classification.json",metrics)
joblib.dump(model,ARTIFACTS/"intent_model.joblib")
train.to_csv(ARTIFACTS/"retrieval_cases.csv",index=False)
agent=SupportAgent(model,train)
test = test.copy(); test["agent_intent"] = pred; test["agent_confidence"] = conf
agent_outputs = [agent.as_dict(message) for message in test.customer_text]
test["decision"] = [row["decision"] for row in agent_outputs]
test["decision_reason"] = [row["reason"] for row in agent_outputs]
test["draft_reply"] = [row["draft_reply"] for row in agent_outputs]
test["evidence_json"] = [json.dumps(row["evidence"]) for row in agent_outputs]
stratified_candidates(test,Path("data")/"golden_candidates.csv")
demo=agent.as_dict(test.customer_text.iloc[0]); write_json(RESULTS/"agent_demo.json",demo)
print(f"Selected {brand}; train={len(train)}, test={len(test)}. Outputs: results/, artifacts/, data/golden_candidates.csv")

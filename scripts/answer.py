import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib, pandas as pd, json
from src.agent import SupportAgent
p=argparse.ArgumentParser();p.add_argument("--artifacts",default="artifacts");p.add_argument("--message",required=True);a=p.parse_args()
path=Path(a.artifacts); agent=SupportAgent(joblib.load(path/"intent_model.joblib"),pd.read_csv(path/"retrieval_cases.csv"));print(json.dumps(agent.as_dict(a.message),indent=2))

import pandas as pd
from src.data import root_threads, split_by_thread
from src.intents import propose_intent, train_classifier
from src.agent import SupportAgent

def test_taxonomy():
    assert propose_intent("I was charged twice") == "billing_charge"
    assert propose_intent("I cannot log in") == "account_access"

def test_thread_split_and_escalation():
    tweets=pd.DataFrame({"tweet_id":["1","2","3","4"],"parent_id":["","1","","3"]})
    rows=pd.DataFrame({"customer_id":["1","3"],"customer_text":["Where is delivery","I need refund"],"agent_reply":["We will check delivery","We will check refund"],"proposed_intent":["delivery_status","refund_request"]})
    train,test=split_by_thread(rows,tweets,test_size=.5)
    assert not set(train.thread_id)&set(test.thread_id)
    model=train_classifier(rows.customer_text,rows.proposed_intent)
    result=SupportAgent(model,rows).answer("I was charged twice")
    assert result.decision=="ESCALATE"

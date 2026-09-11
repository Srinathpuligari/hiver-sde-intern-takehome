"""Dataset selection, thread-safe splits, and reproducible candidate sampling."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .config import RANDOM_STATE

def clean_text(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def load_twcs(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=limit)
    required = {"tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Input lacks TWCS columns: {sorted(missing)}")
    df["tweet_id"] = df.tweet_id.astype(str)
    df["parent_id"] = df.in_response_to_tweet_id.fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    df["text"] = df.text.map(clean_text)
    return df[df.text.str.len().gt(2)].copy()

def customer_agent_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """An agent outbound reply whose direct parent is an inbound customer tweet."""
    customers = df[df.inbound.eq(True)][["tweet_id", "author_id", "text"]].rename(columns={"tweet_id":"customer_id","author_id":"customer_author","text":"customer_text"})
    agents = df[df.inbound.eq(False)][["tweet_id", "author_id", "parent_id", "text"]].rename(columns={"tweet_id":"agent_id","author_id":"brand","text":"agent_reply"})
    pairs = agents.merge(customers, left_on="parent_id", right_on="customer_id", how="inner")
    return pairs[pairs.customer_text.str.len().gt(3)].drop_duplicates("customer_id")

def choose_brand(pairs: pd.DataFrame, min_pairs: int = 100) -> tuple[str, pd.DataFrame]:
    stats = pairs.groupby("brand").agg(pairs=("customer_id","size"), mean_chars=("customer_text", lambda s: round(s.str.len().mean(),1))).sort_values("pairs", ascending=False)
    eligible = stats[stats.pairs.ge(min_pairs)]
    if eligible.empty: raise ValueError("No brand has enough direct customer-agent pairs in this input.")
    return str(eligible.index[0]), stats

def root_threads(df: pd.DataFrame) -> dict[str,str]:
    parent = dict(zip(df.tweet_id, df.parent_id))
    roots = {}
    for tid in parent:
        seen, cur = set(), tid
        while parent.get(cur) and parent[cur] in parent and cur not in seen:
            seen.add(cur); cur = parent[cur]
        roots[tid] = cur
    return roots

def split_by_thread(rows: pd.DataFrame, all_tweets: pd.DataFrame, test_size=.2) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows = rows.copy(); rows["thread_id"] = rows.customer_id.map(root_threads(all_tweets))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_i, test_i = next(splitter.split(rows, groups=rows.thread_id))
    train, test = rows.iloc[train_i].copy(), rows.iloc[test_i].copy()
    if set(train.thread_id) & set(test.thread_id): raise AssertionError("Thread leakage detected")
    return train, test

def stratified_candidates(test: pd.DataFrame, path: Path, n=200) -> pd.DataFrame:
    # Labels here are machine proposals for efficient annotation, not gold labels.
    per = max(1, n // max(1, test.proposed_intent.nunique()))
    out = test.groupby("proposed_intent", group_keys=False).apply(lambda x: x.sample(min(per, len(x)), random_state=RANDOM_STATE), include_groups=True)
    out = out.head(n).copy()
    for col in ["gold_intent","human_auto_handle","correctness","groundedness","completeness","safety","tone"]: out[col] = ""
    out.to_csv(path, index=False); return out

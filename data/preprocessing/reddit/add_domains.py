import json
from pathlib import Path
import pandas as pd

def load_subreddit_to_domain(json_path: Path) -> dict[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    return {
        subreddit.lower(): cluster["cluster"]
        for cluster in clusters
        for subreddit in cluster["subreddits"]
    }

parquet_path = Path("train-00000-of-00001(1).parquet")
domains_json = Path("subreddits_clusters_chatgpt.json")
output_path = Path("train-00000-of-00001-domains.parquet")

df = pd.read_parquet(parquet_path)

subreddit_to_domain = load_subreddit_to_domain(domains_json)

df["domain"] = (
    df["subreddit"]
    .str.lower()
    .map(subreddit_to_domain)
    .fillna("unknown")
)

df.to_parquet(output_path, index=False)
import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import random
from typing import Any
import polars as pl
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

import numpy as np
from collections import defaultdict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

MIN_TEXT_LENGTH = 25
MAX_TEXT_LENGTH = 1000
LANGUAGES = ["de"]

MIN_POSTS_PER_USER = 2
MAX_POSTS_PER_USER = 1000
REMOVE_URL_POSTS = True

SEED=42

@dataclass(frozen=True)
class Post:
    user_id: str
    text: str
    subreddit: str
    title: str
    text_length: int
    domain: str


    def to_serializable(self) -> dict[str, Any]:
        return {"user_id": self.user_id, "text": self.text, "subreddit": self.subreddit, "title": self.title, "text_length": self.text_length, "domain": self.domain}


@dataclass(frozen=True)
class PostPair:
    post_a: Post
    post_b: Post

    @property
    def label(self) -> bool:
        return self.post_a.user_id == self.post_b.user_id

    def to_serializable(self) -> dict[str, Any]:
        return {
            "post_a": self.post_a.to_serializable(),
            "post_b": self.post_b.to_serializable(),
            "label": self.label,
        }


@dataclass(frozen=True)
class AVDataset:
    post_pairs: list[PostPair]

    def to_serializable(self) -> list[dict[str, Any]]:
        return [post_pair.to_serializable() for post_pair in self.post_pairs]
    
def profile_based_dataset_token_ratio_len_dist(
    usernames: list[str],
    data: pl.DataFrame,
    min_tokens: int = 25,
    min_ratio: float = 0.3,
    max_ratio: float = 0.7,
    num_diff_buckets: int = 10,
) -> AVDataset:
    """
    Create a profile-based AV dataset using token-based splits and make sure len difference distribution is similar for positive and negative pairs.
    """

    def get_split_segment(segments: list[str]) -> tuple[str, str]:
        if len(segments) == 2:
            return segments[0], segments[1]

        tokens = [len(m.split()) for m in segments]
        total_tokens = sum(tokens)
        split_point = None

        for _ in range(10):
            target = random.uniform(min_ratio, max_ratio) * total_tokens
            acc = 0
            for i, t in enumerate(tokens[:-1]):
                acc += t
                left = acc
                right = total_tokens - acc
                if acc >= target and left >= min_tokens and right >= min_tokens:
                    split_point = i + 1
                    break
            if split_point is not None:
                break

        if split_point is None:
            split_point = len(segments) // 2

        part_a = " <POST> ".join(segments[:split_point])
        part_b = " <POST> ".join(segments[split_point:])
        return part_a, part_b

    logging.info("Creating profile-based dataset with length-matched negatives...")

    user_texts = {}
    for username in usernames:
        posts = data.filter(pl.col("username") == username)[["text"]].to_dict()["text"][0]
        if len(posts) < 2:
            continue
        random.shuffle(posts)
        user_texts[username] = posts


    positives = []

    for username, segments in user_texts.items():
        part_a, part_b = get_split_segment(segments)

        len_a = len(part_a.split())
        len_b = len(part_b.split())

        post_a = Post(
            user_id=username,
            text=part_a,
            title="",
            subreddit="",
            text_length=len_a,
            domain=""
        )
        post_b = Post(
            user_id=username,
            text=part_b,
            title="",
            subreddit="",
            text_length=len_b,
            domain=""
        )

        length_diff = abs(len_a - len_b) / (len_a + len_b)

        positives.append({
            "pair": PostPair(post_a, post_b),
            "user": username,
            "len_a": len_a,
            "len_b": len_b,
            "length_diff": length_diff,
        })

    # build buckets to store different len difs
    diffs = np.array([p["length_diff"] for p in positives])
    quantiles = np.quantile(diffs, np.linspace(0, 1, num_diff_buckets + 1))

    def diff_bucket(x: float) -> int:
        idx = int(np.searchsorted(quantiles, x, side="right") - 1)
        return max(0, min(num_diff_buckets - 1, idx))

    pos_buckets = defaultdict(list)
    for p in positives:
        b = diff_bucket(p["length_diff"])
        pos_buckets[b].append(p)


    # collect and store all candidates
    all_halves = []

    for username, segments in user_texts.items():
        part_a, part_b = get_split_segment(segments)
        for part in (part_a, part_b):
            all_halves.append({
                "username": username,
                "text": part,
                "len": len(part.split()),
            })

    #sample to match distribution
    negative_pairs = []

    for bucket_id, bucket_positives in pos_buckets.items():
        lo = quantiles[bucket_id]
        hi = quantiles[bucket_id + 1]

        for p in bucket_positives:
            post_a = p["pair"].post_a
            len_a = p["len_a"]

            candidates = []
            for h in all_halves:
                if h["username"] == post_a.user_id:
                    continue
                diff = abs(len_a - h["len"]) / (len_a + h["len"])
                if lo <= diff <= hi:
                    candidates.append(h)

            # fallback
            if not candidates:
                candidates = [
                    h for h in all_halves if h["username"] != post_a.user_id
                ]

            h = random.choice(candidates)

            post_other = Post(
                user_id=h["username"],
                text=h["text"],
                title="",
                subreddit="",
                text_length=h["len"],
                domain=""
            )

            negative_pairs.append(PostPair(post_a, post_other))

    logging.info(
        f"Created {len(positives)} positives and "
        f"{len(negative_pairs)} negatives with matched length distribution."
    )

    # final
    return AVDataset(
        [p["pair"] for p in positives] + negative_pairs
    )


def users_to_dataset(usernames: list[str], data: pl.DataFrame, domain_mode: str, max_per_user: int = 2) -> AVDataset:
    """Take list of users for a split and original data to form a dataset. Dataset is a list of sample pairs. Can be used for in-domain and cross-domain pair sampling."""
    positive_pairs = []
    negative_pairs = []

    logging.info("Collecting positive pairs")
    
    for username in usernames:
        user_posts= data.filter(pl.col("username") == username)[
            ["text",
            "title",
            "subreddit",
            "text_length",
            "domain"]
        ].to_dict()


        user_texts = user_posts["text"].to_list()[0]
        user_titles = user_posts["title"].to_list()[0]
        user_subreddits = user_posts["subreddit"].to_list()[0]
        user_text_lengths = user_posts["text_length"].to_list()[0]
        user_domains = user_posts["domain"].to_list()[0]
        user_posts = list(zip(user_texts, user_titles, user_subreddits, user_text_lengths, user_domains))

        # Keep either combinations with domain1 = domain1 or with domain1 != domain1
        all_combos = []
        for a, b in combinations(user_posts, 2):
            domain_a = a[4]
            domain_b = b[4]

            if domain_mode == "in_domain" and domain_a == domain_b:
                all_combos.append((a, b))
            elif domain_mode == "cross_domain" and domain_a != domain_b:
                all_combos.append((a, b))

        if len(all_combos) <= max_per_user:
            sampled = all_combos
        else:
            sampled = random.sample(all_combos, max_per_user)

        for _post_a, _post_b in sampled:
            post_a = Post(user_id=username, text=_post_a[0], title=_post_a[1], subreddit=_post_a[2], text_length=_post_a[3], domain=_post_a[4])
            post_b = Post(user_id=username, text=_post_b[0], title=_post_b[1], subreddit=_post_b[2], text_length=_post_b[3], domain=_post_b[4])
            post_pair = PostPair(post_a, post_b)
            positive_pairs.append(post_pair)
    logging.info(f"{len(positive_pairs)} positive pairs.")

    logging.info("Collecting negative pairs")
    while len(negative_pairs) < len(positive_pairs):
        _user_a, _user_b = random.sample(usernames, 2)

        posts_a = data.filter(pl.col("username") == _user_a)[["text","title","subreddit","text_length","domain"]].to_dict()
        posts_b = data.filter(pl.col("username") == _user_b)[["text","title","subreddit","text_length","domain"]].to_dict()

        user_posts_a = list(zip(
            posts_a["text"][0], posts_a["title"][0], posts_a["subreddit"][0], posts_a["text_length"][0], posts_a["domain"][0]
        ))
        user_posts_b = list(zip(
            posts_b["text"][0], posts_b["title"][0], posts_b["subreddit"][0], posts_b["text_length"][0], posts_b["domain"][0]
        ))

        # filter according to domain_mode
        valid_pairs = []
        for a in user_posts_a:
            for b in user_posts_b:
                if domain_mode == "in_domain" and a[4] == b[4]:
                    valid_pairs.append((a, b))
                elif domain_mode == "cross_domain" and a[4] != b[4]:
                    valid_pairs.append((a, b))

        if not valid_pairs:
            continue  # skip if no valid pair found

        _post_a, _post_b = random.choice(valid_pairs)

        post_a = Post(user_id=_user_a, text=_post_a[0], title=_post_a[1], subreddit=_post_a[2], text_length=_post_a[3], domain=_post_a[4])
        post_b = Post(user_id=_user_b, text=_post_b[0], title=_post_b[1], subreddit=_post_b[2], text_length=_post_b[3], domain=_post_b[4])
        negative_pairs.append(PostPair(post_a, post_b))

    logging.info("Collecting negative pairs")

    logging.info(f"{len(negative_pairs)} negative pairs.")

    print("Final size:", len(positive_pairs) + len(negative_pairs))
    return AVDataset(positive_pairs + negative_pairs)




def main():
    parser = ArgumentParser()
    parser.add_argument("--in_file", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    input_file = Path(args.in_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)

    # read in file and preprocessing
    df = pl.read_parquet(input_file)
    logging.info(f"Original # of posts: {len(df)}")
    df = df.filter(pl.col("domain") != "unknown")
    logging.info(f"# of posts with assigned domain: {len(df)}")

    # Keep posts with no URLs
    logging.info("Removing posts with URLs.")
    if REMOVE_URL_POSTS:
        df = df.filter(pl.col("text_url_count").eq(0))

    # Filter by length and langs
    logging.info("Filtering by length and language.")
    df = df.filter(
        pl.col("text_length").gt(MIN_TEXT_LENGTH),
        pl.col("text_length").lt(MAX_TEXT_LENGTH),
        pl.col("text_lang").is_in(LANGUAGES),
    )
    logging.info(f"Filtered # of posts: {len(df)}")

    # Group posts by users
    logging.info("Grouping by usernames...")
    grouped_df = df.group_by("username").agg([
        pl.col("text"),
        pl.col("title"),
        pl.col("subreddit"),
        pl.col("text_length"),
        pl.col("domain"),
        ])

    logging.info(f"{len(grouped_df)} unique users.")
    logging.info(
        f"Removing users with fewer than {MIN_POSTS_PER_USER} "
        f"or more than {MAX_POSTS_PER_USER} posts"
    )
    grouped_df = grouped_df.filter(
        pl.col("text").list.len().ge(MIN_POSTS_PER_USER),
        pl.col("text").list.len().le(MAX_POSTS_PER_USER),
    )

    # Remove users with bot-like names
    bot_usernames = ["null", "AutoModerator"]
    grouped_df = grouped_df.filter(
        ~(pl.col("username").str.contains_any(["bot", "Bot", "BOT"] + bot_usernames))
    )

    logging.info(f"{len(grouped_df)} unique users after filtering.")

    usernames = list(grouped_df["username"])
    train_users, val_test_users = train_test_split(usernames, test_size=0.4, random_state=SEED)
    val_users, test_users = train_test_split(val_test_users, test_size=0.5, random_state=SEED)

    logging.info("6/2/2 split to users")

    logging.info(
        f"train/validation/test: {len(train_users)}/{len(val_users)}/{len(test_users)}"
    )



    # create AV datasets

    profile_datasets = {}
    in_domain_datasets = {}
    cross_domain_datasets = {}

    for split_name, split_users in [
        ("train", train_users),
        ("validation", val_users),
        ("test", test_users),
    ]:
        logging.info(f"Processing {split_name} split")

        # Profile-based
        profile_dataset = profile_based_dataset_token_ratio_len_dist(
            split_users,
            grouped_df,
        )

        profile_datasets[split_name] = Dataset.from_list(
            profile_dataset.to_serializable()
        )

        logging.info(
            f"{split_name}_profile_based: "
            f"{len(profile_dataset.post_pairs)} pairs"
        )

        # In-domain
        in_domain_dataset = users_to_dataset(
            split_users,
            grouped_df,
            domain_mode="in_domain",
        )

        in_domain_datasets[split_name] = Dataset.from_list(
            in_domain_dataset.to_serializable()
        )

        logging.info(
            f"{split_name}_in_domain: "
            f"{len(in_domain_dataset.post_pairs)} pairs"
        )

        # cross-domain
        cross_domain_dataset = users_to_dataset(
            split_users,
            grouped_df,
            domain_mode="cross_domain",
        )

        cross_domain_datasets[split_name] = Dataset.from_list(
            cross_domain_dataset.to_serializable()
        )

        logging.info(
            f"{split_name}_cross_domain: "
            f"{len(cross_domain_dataset.post_pairs)} pairs"
        )


    # Convert to DatasetDicts
    profile_dataset_dict = DatasetDict(profile_datasets)
    in_domain_dataset_dict = DatasetDict(in_domain_datasets)
    cross_domain_dataset_dict = DatasetDict(cross_domain_datasets)


    # Save each dataset
    profile_dataset_dict.save_to_disk(
        args.out_dir / "profile_based"
    )

    in_domain_dataset_dict.save_to_disk(
        args.out_dir / "in_domain"
    )

    cross_domain_dataset_dict.save_to_disk(
        args.out_dir / "cross_domain"
    )

    logging.info(
        f"Saved profile-based dataset to "
        f"{args.out_dir / 'profile_based'}"
    )

    logging.info(
        f"Saved in-domain dataset to "
        f"{args.out_dir / 'in_domain'}"
    )

    logging.info(
        f"Saved cross-domain dataset to "
        f"{args.out_dir / 'cross_domain'}"
    )
    

if __name__ == "__main__":
    main()

import logging
import random
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
from typing import Any
from datasets import Dataset, DatasetDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


MIN_WORDS = 25
MAX_CHAR_LENGTH = 300

POS_SAMPLES_PER_AUTHOR = 2
NEG_SAMPLES_PER_AUTHOR = 2

MIN_TEXTS_PER_AUTHOR = 2

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

SEED = 42


def build_pairs(
    target_authors: list[str],
    author_texts: dict[str, list[str]],
    pos_per_author: int = POS_SAMPLES_PER_AUTHOR,
    neg_per_author: int = NEG_SAMPLES_PER_AUTHOR,
) -> list[dict[str, Any]]:

    pairs = []

    all_authors = list(author_texts.keys())

    for author in target_authors:

        texts = author_texts[author]

        # positive pairs
        for _ in range(pos_per_author):

            if len(texts) < 2:
                continue

            text_a, text_b = random.sample(texts, 2)

            pairs.append(
                {
                    "post_a": {
                        "user_id": author,
                        "text": text_a,
                    },
                    "post_b": {
                        "user_id": author,
                        "text": text_b,
                    },
                    "label": True,
                }
            )

        # negative pairs
        other_authors = [
            a for a in all_authors
            if a != author
        ]

        if not other_authors:
            continue

        for _ in range(neg_per_author):

            neg_author = random.choice(other_authors)

            text_a = random.choice(texts)
            text_b = random.choice(author_texts[neg_author])

            pairs.append(
                {
                    "post_a": {
                        "user_id": author,
                        "text": text_a,
                    },
                    "post_b": {
                        "user_id": neg_author,
                        "text": text_b,
                    },
                    "label": False,
                }
            )

    return pairs


def main():

    parser = ArgumentParser()
    parser.add_argument("--in_file", type=Path, required=True) #path to the filter_twitter.py output csv
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    args.out_dir.mkdir(parents=True, exist_ok=True,)

    # load data

    logging.info(f"Loading {args.in_file}")

    df = pd.read_csv(args.in_file)

    logging.info(
        f"Original samples: {len(df)}"
    )


    # filtering
    df["text"] = (df["text"].astype(str).str.strip())

    df = df[df["text"].str.split().str.len() >= MIN_WORDS]

    logging.info(
        f"After min word filtering: {len(df)}"
    )


    df = df[df["text"].str.len() <= MAX_CHAR_LENGTH]

    logging.info(
        f"After max length filtering: {len(df)}"
    )


    # group by author

    author_texts = defaultdict(list)

    for user_id, group in df.groupby("user_id"):
        author_texts[user_id] = group["text"].tolist()


    author_texts = {
        author: texts
        for author, texts in author_texts.items()
        if len(texts) >= MIN_TEXTS_PER_AUTHOR
    }


    authors = list(author_texts.keys())

    logging.info(
        f"Authors after filtering: {len(authors)}"
    )


    # split authors
    train_authors, temp_authors = train_test_split(
        authors,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=SEED,
    )

    val_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    val_authors, test_authors = train_test_split(
        temp_authors,
        test_size=val_test_ratio,
        random_state=SEED,
    )


    splits = {
        "train": train_authors,
        "validation": val_authors,
        "test": test_authors,
    }


    logging.info(
        f"Split sizes: "
        f"train={len(train_authors)}, "
        f"validation={len(val_authors)}, "
        f"test={len(test_authors)}"
    )


    # create datasets for each split

    datasets = {}

    for split_name, split_authors in splits.items():

        logging.info(
            f"Creating {split_name} dataset"
        )

        pairs = build_pairs(
            split_authors,
            author_texts,
        )

        datasets[split_name] = Dataset.from_list(pairs)

        logging.info(
            f"{split_name}: {len(pairs)} pairs"
        )

    dataset_dict = DatasetDict(datasets)

    dataset_dict.save_to_disk(args.out_dir)

    logging.info(
        f"Saved dataset to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
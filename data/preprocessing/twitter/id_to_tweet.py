import argparse
import gzip
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any
from datasets import DatasetDict, load_from_disk


URL_PATTERN = re.compile(r"https?://\S+")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub("", text).rstrip()


def load_json_gz_from_zip(zip_path: Path, member: str) -> list[dict[str, Any]]:
    """Load one JSON.GZ file from a ZIP archive."""
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(member) as compressed_file:
            return json.loads(gzip.decompress(compressed_file.read()).decode("utf-8"))


def build_id_to_text_mapping(original_dir: Path) -> dict[str, str]:
    """Build a tweet ID -> original tweet text mapping from ZIP archives."""
    id_to_text: dict[str, str] = {}
    total_tweets = 0
    duplicate_ids = 0
    valid_files = 0
    invalid_files = 0

    zip_files = list(original_dir.rglob("*.zip"))
    logging.info(f"Found {len(zip_files)} ZIP files.")

    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                members = [name for name in z.namelist() if name.lower().endswith(".json.gz")]

                for member in members:
                    try:
                        tweets = load_json_gz_from_zip(zip_path, member)
                        valid_files += 1

                        for tweet in tweets:
                            tweet_id = tweet.get("id")
                            text = tweet.get("text")

                            if tweet_id is None or text is None:
                                continue

                            tweet_id = str(tweet_id)
                            total_tweets += 1

                            if tweet_id in id_to_text:
                                duplicate_ids += 1

                            id_to_text[tweet_id] = remove_urls(text)

                    except Exception as e:
                        invalid_files += 1
                        logging.warning(f"Failed to process {zip_path.name}:{member}: {e}")

        except Exception as e:
            logging.warning(f"Failed to open {zip_path}: {e}")

    logging.info(
        f"Mapping complete: {total_tweets:,} tweets, {len(id_to_text):,} unique IDs, "
        f"{duplicate_ids:,} duplicates, {valid_files:,} valid files, {invalid_files:,} invalid files."
    )

    return id_to_text


def replace_ids_with_text(dataset, id_to_text: dict[str, str], split: str):
    """Replace tweet IDs in post_a/post_b text fields with the original tweet text."""
    stats = {"total": 0, "found": 0, "missing": 0}

    def replace_entry(entry: dict[str, Any]) -> dict[str, Any]:
        entry = dict(entry)

        for post_key in ("post_a", "post_b"):
            post = entry.get(post_key)

            if post is None or "text" not in post:
                continue

            stats["total"] += 1
            tweet_id = str(post["text"])

            if tweet_id not in id_to_text:
                stats["missing"] += 1
                continue

            stats["found"] += 1
            post = dict(post)
            post["text"] = id_to_text[tweet_id]
            entry[post_key] = post

        return entry

    dataset = dataset.map(replace_entry, desc=f"Processing {split}")

    logging.info(
        f"{split}: {stats['found']:,}/{stats['total']:,} IDs replaced, "
        f"{stats['missing']:,} missing."
    )

    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", type=Path, required=True, help="Directory containing the original ZIP files.")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Local HF dataset containing train, test and validation split with text IDs.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for the text dataset.")
    args = parser.parse_args()

    logging.info("Building ID -> text mapping...")
    id_to_text = build_id_to_text_mapping(args.original_dir)

    logging.info(f"Loading dataset from {args.dataset_dir}")
    dataset = load_from_disk(str(args.dataset_dir))

    for split in ("train", "validation", "test"):
        if split in dataset:
            dataset[split] = replace_ids_with_text(dataset[split], id_to_text, split)

    logging.info(f"Saving dataset to {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))
    logging.info("Done.")


if __name__ == "__main__":
    main()


import logging
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import sienna
from langid.langid import LanguageIdentifier, model


logging.getLogger().setLevel(logging.INFO)


#metrics

@dataclass(frozen=True)
class SpaceBasedLength:
    """Count words by splitting text on spaces."""

    def run(self, text: str) -> int:
        return len(text.split())


@dataclass(frozen=True)
class URLCounter:
    """Count URLs starting with http:// or https://."""

    def run(self, text: str) -> int:
        return len(re.findall(r"http(s)?://", text))


@dataclass(frozen=True)
class LangID:
    """Identify the language of a text."""

    identifier = LanguageIdentifier.from_modelstring(
        model,
        norm_probs=True,
    )

    def run(self, text: str) -> tuple[str, float]:
        lang, confidence = self.identifier.classify(text)
        return lang, confidence



def main():
    parser = ArgumentParser()
    parser.add_argument("-input", type=Path, required=True)
    parser.add_argument("-output", type=Path, required=True)
    args = parser.parse_args()

    input_dir: Path = args.input
    output_file: Path = args.output

    space_based_length = SpaceBasedLength()
    langid = LangID()
    url_counter = URLCounter()

    all_posts = []

    # load original json files and filter

    for file_path in input_dir.glob("*.json"):
        tweets = cast(list[dict[str, Any]], sienna.load(file_path))

        logging.info(
            f"{len(tweets)} found in {file_path.name}"
        )

        for tweet in tweets:

            # Skip unwanted tweet types
            if tweet["type"] in (
                "status",
                "reply",
                "user",
                "retweet",
            ):
                continue

            text = tweet["text"]

            # Calculate metrics
            tweet["space_based_length"] = (
                space_based_length.run(text)
            )

            tweet["lang"] = langid.run(text)[0]

            tweet["url_count"] = url_counter.run(text)

            all_posts.append(tweet)

    logging.info(
        f"Total posts after type filtering: {len(all_posts)}"
    )

    # convert to dataframe

    df = pd.DataFrame(all_posts)

    # remove all posts containing urls

    logging.info("Removing posts with URLs.")

    df = df[df["url_count"] == 0]

    logging.info(
        f"Posts after URL filtering: {len(df)}"
    )

    # keep German only

    logging.info("Keeping German posts only.")

    df = df[df["lang"] == "de"]

    logging.info(
        f"Posts after language filtering: {len(df)}"
    )

    # remove bots by username

    logging.info(
        "Removing users with bot-like usernames."
    )

    bot_pattern = r"bot|Bot|BOT"

    df = df[
        ~df["user"].astype(str).str.contains(
            bot_pattern,
            regex=True,
            na=False,
        )
    ]

    logging.info(
        f"Posts after bot filtering: {len(df)}"
    )

    logging.info(
        f"Remaining users: {df['user'].nunique()}"
    )


    df = df.rename(columns={"user": "user_id"})

    # save to csv

    output_file.parent.mkdir(parents=True, exist_ok=True,)

    df.to_csv(output_file, index=False,)

    logging.info(
        f"Saved filtered dataset to {output_file}"
    )


if __name__ == "__main__":
    main()

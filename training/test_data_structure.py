import argparse
from pprint import pprint

from dataset_loader import (
    RedditDatasetLoaderInDomain,
    RedditDatasetLoaderCrossDomain,
    RedditDatasetLoaderProfileBased,
    MixedLoader,
)
from prompters import SimplePrompter


def build_loader(dataset_name: str, prompt_template: str, lang: str = "en"):
    prompter = SimplePrompter(prompt_template)

    if dataset_name == "reddit_in_domain":
        return RedditDatasetLoaderInDomain(prompter)
    elif dataset_name == "reddit_cross_domain":
        return RedditDatasetLoaderCrossDomain(prompter)
    elif dataset_name == "reddit_profile_based":
        return RedditDatasetLoaderProfileBased(prompter)
    elif dataset_name == "mix_reddit_twitter":
        return MixedLoader(prompter)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mix_reddit_twitter",
        choices=[
            "reddit_in_domain",
            "reddit_cross_domain",
            "reddit_profile_based",
            "mix_reddit_twitter",
        ],
    )
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    prompt_template = (
        "Wurden die folgenden zwei Texte von dem selben Autor verfasst?\n"
        "Text A: {text_a}\n"
        "Text B: {text_b}\n"
        "Antworte mit 'Ja' oder 'Nein'.\n"
        "Antwort: "
    )

    loader = build_loader(args.dataset, prompt_template, args.lang)
    ds = loader.load()

    print(f"Loaded dataset: {args.dataset}")
    print(ds)
    print()

    for split in ds.keys():
        print(f"=== Split: {split} | size={len(ds[split])} ===")
        for i in range(min(args.num_samples, len(ds[split]))):
            print(f"\n--- {split}[{i}] ---")
            sample = ds[split][i]

            print("Keys:", list(sample.keys()))

            pprint(sample)

            if "text" in sample:
                print("\n[text field]")
                print(sample["text"])
        print()


if __name__ == "__main__":
    main()
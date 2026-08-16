from dataclasses import dataclass
from typing import Mapping

from datasets import Dataset, load_dataset, load_from_disk

from prompters import SimplePrompter

CLASS2LABEL = {True: "Ja", False: "Nein"}


@dataclass(frozen=True)
class TwitterDatasetLoader:
    prompter: SimplePrompter

    def load(self) -> Dataset | tuple[Dataset, list[bool]]:
        ds = load_dataset("your twitter dataset name here")

        def preprocess_dataset(example: Mapping):
            content = self.prompter.run(
                example["post_a"]["text"], example["post_b"]["text"]
            )

            chat = [
                {"role": "user", "content": str(content)},
                {"role": "assistant", "content": CLASS2LABEL[example["label"]]},
            ]
            return {"messages": chat}

        ds = ds.map(
            preprocess_dataset,
            batched=False,
            remove_columns=ds["train"].column_names,
        )

        return ds


@dataclass(frozen=True)
class RedditDatasetLoaderInDomain:
    prompter: SimplePrompter

    def load(self) -> Dataset | tuple[Dataset, list[bool]]:
        ds = load_dataset("your reddit in-domain dataset name here")

        def preprocess_dataset(example: Mapping):
            content = self.prompter.run(
                example["post_a"]["text"], example["post_b"]["text"]
            )

            chat = [
                {"role": "user", "content": str(content)},
                {"role": "assistant", "content": CLASS2LABEL[example["label"]]},
            ]
            return {"messages": chat}

        ds = ds.map(
            preprocess_dataset,
            batched=False,
            remove_columns=ds["train"].column_names,
        )

        return ds

    
@dataclass(frozen=True)
class RedditDatasetLoaderCrossDomain:
    prompter: SimplePrompter

    def load(self) -> Dataset | tuple[Dataset, list[bool]]:
        ds = load_dataset("your reddit cross-domain dataset name here")
        def preprocess_dataset(example: Mapping):
            content = self.prompter.run(
                example["post_a"]["text"], example["post_b"]["text"]
            )

            chat = [
                {"role": "user", "content": str(content)},
                {"role": "assistant", "content": CLASS2LABEL[example["label"]]},
            ]
            return {"messages": chat}

        ds = ds.map(
            preprocess_dataset,
            batched=False,
            remove_columns=ds["train"].column_names,
        )

        return ds


@dataclass(frozen=True)
class RedditDatasetLoaderProfileBased:
    prompter: SimplePrompter

    def load(self) -> Dataset | tuple[Dataset, list[bool]]:
        ds = load_dataset("your reddit profile-based dataset name here")
        def preprocess_dataset(example: Mapping):
            content = self.prompter.run(
                example["post_a"]["text"], example["post_b"]["text"]
            )

            chat = [
                {"role": "user", "content": str(content)},
                {"role": "assistant", "content": CLASS2LABEL[example["label"]]},
            ]
            return {"messages": chat}

        ds = ds.map(
            preprocess_dataset,
            batched=False,
            remove_columns=ds["train"].column_names,
        )

        return ds

    
@dataclass(frozen=True)
class MixedLoader:
    prompter: SimplePrompter

    def load(self) -> Dataset | tuple[Dataset, list[bool]]:
        ds = load_dataset("your mixed dataset name here")
        def preprocess_dataset(example: Mapping):
            content = self.prompter.run(
                example["post_a"]["text"], example["post_b"]["text"]
            )

            chat = [
                {"role": "user", "content": str(content)},
                {"role": "assistant", "content": CLASS2LABEL[example["label"]]},
            ]
            return {"messages": chat}

        ds = ds.map(
            preprocess_dataset,
            batched=False,
            remove_columns=ds["train"].column_names,
        )

        return ds

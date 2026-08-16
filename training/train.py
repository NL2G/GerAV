import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, NewType, Sequence

import sienna
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from dataset_loader import (
    TwitterDatasetLoader,
    RedditDatasetLoaderInDomain,
    RedditDatasetLoaderCrossDomain,
    RedditDatasetLoaderProfileBased,
    MixedLoader
)
from prompters import SimplePrompter
from utils import read_toml

Hparams = NewType("Hparams", Mapping[str, Any])


def hparams_to_dir(hparams: Hparams) -> Path:
    sorted_key_values = sorted(hparams.items(), key=lambda itm: itm[0])
    return Path("-".join([f"{k}:{v}" for k, v in sorted_key_values]))


@dataclass(frozen=True)
class Config:
    model_name: str
    tokenizer_name: str
    output_dir: Path

    prompt_template: str

    dataset: str

    batch_size: int
    num_train_epochs: int
    tuning_seed: int

    learning_rates: Sequence[float]
    weight_decays: Sequence[float]
    seeds: Sequence[int]

    do_lora: bool
    lora_rs: Sequence[int] | None
    lora_alphas: Sequence[int] | None
    lora_dropouts: Sequence[float] | None

    lang: str

    def __post_init__(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.do_lora:
            if (
                self.lora_rs is None
                or self.lora_alphas is None
                or self.lora_dropouts is None
            ):
                raise ValueError("You have to set `lora_r` for lora fine-tuning.")

    @classmethod
    def from_cli(cls) -> "Config":
        parser = ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        args = parser.parse_args()

        _config = read_toml(args.config)

        return cls(
            model_name=_config["model_name"],
            tokenizer_name=_config["tokenizer_name"],
            output_dir=Path(_config["output_dir"]),
            prompt_template=_config["prompt_template"],
            dataset=_config["dataset"],
            batch_size=_config["batch_size"],
            num_train_epochs=_config["num_train_epochs"],
            tuning_seed=_config["tuning_seed"],
            lang=_config["lang"],
            learning_rates=_config["learning_rates"],
            weight_decays=_config["weight_decays"],
            seeds=_config["seeds"],
            do_lora=_config["do_lora"],
            lora_rs=_config.get("lora_rs", None),
            lora_alphas=_config.get("lora_alphas", None),
            lora_dropouts=_config.get("lora_dropouts", None),
        )

    def get_hparams_combinations(self) -> Sequence[Hparams]:
        hparams_li: Sequence[Hparams] = []
        if self.do_lora:
            for lr, wd, r, al, do in product(
                self.learning_rates,
                self.weight_decays,
                self.lora_rs,
                self.lora_alphas,
                self.lora_dropouts,
            ):
                hparams_li.append(
                    {
                        "learning_rate": lr,
                        "weight_decay": wd,
                        "lora_r": r,
                        "lora_alpha": al,
                        "lora_dropout": do,
                    }
                )
        else:
            for lr, wd in product(self.learning_rates, self.weight_decays):
                hparams_li.append({"learning_rate": lr, "weight_decay": wd})
        return hparams_li


@dataclass(frozen=True)
class TrainingResult:
    hparams: Hparams
    validation_metrics: dict[str, float]

    def to_serializable(self) -> Mapping[str, Hparams | dict[str, float]]:
        return {
            "hparams": self.hparams,
            "validation_metrics": self.validation_metrics,
        }


@dataclass
class TrainingResults:
    results: list[TrainingResult]

    @classmethod
    def init(cls, path: Path) -> "TrainingResults":
        """Load from path if exists, initialize with empty result if does not exist

        Args:
            path: path to save (or load) the results

        Returns:
            TrainingResults
        """
        if path.exists():
            logging.info(
                f"{path} exists already, so we are loading and resuming the process."
            )
            results = [
                TrainingResult(
                    hparams=r["hparams"], validation_metrics=r["validation_metrics"]
                )
                for r in sienna.load(path)["results"]
            ]
        else:
            results = []
        return cls(results=results)

    @property
    def best_result(self) -> TrainingResult:
        best_result = sorted(
            self.results,
            key=lambda result: result.validation_metrics["eval_loss"],
        )[0]
        return best_result

    def to_serializable(self) -> dict[str, Any]:
        return {"results": [r.to_serializable() for r in self.results]}

    def save(self, path: Path) -> None:
        sienna.save(self.to_serializable(), path)


@dataclass(frozen=True)
class Runner:
    config: Config

    @classmethod
    def from_cli(cls) -> "Runner":
        config = Config.from_cli()
        return cls(config=config)

    def load_model(self, hparams: Hparams) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype="auto",
            attn_implementation="eager",
        )
        if self.config.do_lora:
            # create LoRA configuration object
            logging.info("Using LoRA adapter for fine-tuning.")
            logging.info(
                f"lora_r: {hparams['lora_r']}. "
                f"lora_alpha: {hparams['lora_alpha']} "
                f" lora_dropout: {hparams['lora_dropout']}"
            )
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,  # type of task to train on
                inference_mode=False,  # set to False for training
                r=hparams["lora_r"],  # dimension of the smaller matrices
                lora_alpha=hparams["lora_alpha"],  # scaling factor
                lora_dropout=hparams["lora_dropout"],  # dropout of LoRA layers
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        return model

    def load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_dataset(self) -> Dataset:
        prompter = SimplePrompter(self.config.prompt_template)

        if self.config.dataset == "twitter":
            dataset_loader = TwitterDatasetLoader(prompter)

        elif self.config.dataset == "reddit_in_domain":
            dataset_loader = RedditDatasetLoaderInDomain(prompter)
        elif self.config.dataset == "reddit_cross_domain":
            dataset_loader = RedditDatasetLoaderCrossDomain(prompter)
        elif self.config.dataset == "reddit_profile_based":
            dataset_loader = RedditDatasetLoaderProfileBased(prompter)
        elif self.config.dataset == "mix_reddit_twitter":
            dataset_loader = MixedLoader(prompter)
        else:
            raise ValueError(f"{self.config.dataset} dataset does not exist.")

        ds = dataset_loader.load()

        return ds

    def train(
        self, hparams: Hparams, seed: int, output_dir: Path
    ) -> tuple[AutoModelForCausalLM, TrainingResult]:
        torch.cuda.empty_cache()
        logging.info("Starting one training")
        logging.info(f"Hyperparameters: {hparams}")

        logging.info(f"Setting seed: {seed}")
        set_seed(seed)

        model = self.load_model(hparams)
        # tokenizer = self.load_tokenizer()
        ds = self.load_dataset()

        print(f"Number of training samples: {len(ds['train'])}")

        training_args = SFTConfig(
            output_dir=str(output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            push_to_hub=False,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            load_best_model_at_end=True,
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            args=training_args,
        )
        trainer.train()

        best_checkpoint_model_metrics = trainer.evaluate()

        return model, TrainingResult(
            hparams=hparams,
            validation_metrics=best_checkpoint_model_metrics,
        )

    def fine_best_hparams(self) -> Hparams:
        combinations = self.config.get_hparams_combinations()
        print("Hyperparameter combinations:", combinations)
        return combinations[0]

    def run(self) -> None:
        best_hparams = self.fine_best_hparams()
        for seed in self.config.seeds:
            output_dir = self.config.output_dir / "final_models" / f"seed:{seed}"
            model, _ = self.train(best_hparams, seed, output_dir)
            logging.info(
                f"Save best model with seed: {seed} at {output_dir / 'best_model'}"
            )
            model.save_pretrained(output_dir / "best_model")


def main():
    runner = Runner.from_cli()
    runner.run()


if __name__ == "__main__":
    main()

# LLM-based Authorship Verification

Code for the fine-tuning experiments from GerAV.

## Datasets

Dataset names and paths are not included in this repository. The placeholders in `dataset_loader.py` need to be replaced with the corresponding dataset sources.

## Training

Training is based on Hugging Face Transformers and TRL, with optional LoRA fine-tuning using PEFT.

```bash
python -m llm4av.simple_tuning.train --config <config-file>
````

## Hyperparameters
As the result of hyperparameter search, we use the best configuration found in an initial validation experiment.

## Notes

Configuration files may need to be adjusted to the local environment. The datasets themselves are not included.


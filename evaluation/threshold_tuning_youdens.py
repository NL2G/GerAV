import argparse
import gc
import sklearn
import torch
from datasets import load_from_disk, load_dataset
import numpy as np

from GerAV import GerAV
from MStyleDistance import MStyleDistance
from MultilingualStyleRepresentation import MultilingualStyleRepresentation
from PPMEval import PPM
from SbertEval import SBERT
from AdhominemEval import ADHOMINEM
from FeatureDifferenceEval import FeatureDifference




MODEL_DICT =  {
    "gpt-5": (GerAV, [], {"gpt_mode": True, "base_path": "./lora_configs/configs_mix", "baseline": True, "tuned_model":"gpt-5"}),
    "gpt-5-lip": (GerAV, [], {"gpt_mode": True, "base_path": "./lora_configs/configs_mix", "baseline": True, "tuned_model":"gpt-5", "use_lip": True}),
    "gpt-5-lip-ger": (GerAV, [], {"gpt_mode": True, "base_path": "./lora_configs/configs_mix", "baseline": True, "tuned_model":"gpt-5", "use_lip_ger": True}),
    "gpt-5-zero-fast": (GerAV, [], {"gpt_mode": True, "base_path": "./lora_configs/configs_mix", "baseline": True, "tuned_model":"gpt-5", "use_zero_fast": True}),

    "m_style_distance_mixed": (MStyleDistance, []),
    "msr_mixed": (MultilingualStyleRepresentation, []),
    
    "ngram_twitter": (FeatureDifference, [], {"model_type": "twitter"}),
    "ngram_reddit_in_domain": (FeatureDifference, [], {"model_type": "in"}),
    "ngram_reddit_cross_domain": (FeatureDifference, [], {"model_type": "cross"}),
    "ngram_reddit_profile_based": (FeatureDifference, [], {"model_type": "profile"}),
    "ngram_mixed": (FeatureDifference, [], {"model_type": "mixed"}),

    "ppm_twitter": (PPM, [], {"model_type": "twitter"}),
    "ppm_reddit_in_domain": (PPM, [], {"model_type": "in"}),
    "ppm_reddit_cross_domain": (PPM, [], {"model_type": "cross"}),
    "ppm_reddit_profile_based": (PPM, [], {"model_type": "profile"}),
    "ppm_mixed": (PPM, [], {"model_type": "mixed"}),

    "sbert_twitter": (SBERT, [], {"model_type": "twitter"}),
    "sbert_reddit_in_domain": (SBERT, [], {"model_type": "in"}),
    "sbert_reddit_cross_domain": (SBERT, [], {"model_type": "cross"}),
    "sbert_reddit_profile_based": (SBERT, [], {"model_type": "profile"}),
    "sbert_mixed": (SBERT, [], {"model_type": "mixed"}),

    "adhominem_twitter": (ADHOMINEM, [], {"model_type": "twitter"}),
    "adhominem_reddit_in_domain": (ADHOMINEM, [], {"model_type": "in"}),
    "adhominem_reddit_cross_domain": (ADHOMINEM, [], {"model_type": "cross"}),
    "adhominem_reddit_profile_based": (ADHOMINEM, [], {"model_type": "profile"}),
    "adhominem_mixed": (ADHOMINEM, [], {"model_type": "mixed"}),
    
    "baseline_gemma_3_12b_it": (GerAV, [], {"base_path": "./lora_configs/twitter","tuned_model": "gemma-3-12b-it", "tune_dataset": "twitter", "seed": 42, "baseline": True, "custom_threshold": None}),
    "baseline_llama-3.1-8b-instruct": (GerAV, [], {"base_path": "./lora_configs/twitter","tuned_model": "llama-3.1-8b-instruct", "tune_dataset": "twitter", "seed": 42, "baseline": True, "custom_threshold": None}),
    "baseline_llama-3.2-3b-instruct": (GerAV, [], {"base_path": "./lora_configs/twitter","tuned_model": "llama-3.2-3b-instruct", "tune_dataset": "twitter", "seed": 42, "baseline": True, "custom_threshold": None}),
    "baseline_qwen-2.5-7b-instruct": (GerAV, [], {"base_path": "./lora_configs/twitter","tuned_model": "qwen-2.5-7b-instruct", "tune_dataset": "twitter", "seed": 42, "baseline": True, "custom_threshold": None}),
    "baseline_lammlein": (GerAV, [], {"base_path": "./lora_configs/twitter","tuned_model": "lammlein", "tune_dataset": "twitter", "seed": 42, "baseline": True, "custom_threshold": None}),
    
    "gerav___llama-3.2-3b-instruct___twitter": (GerAV, [], {"base_path": "./lora_configs/twitter", "tuned_model": "llama-3.2-3b-instruct", "tune_dataset": "twitter", "seed": 42, "custom_threshold": None}),
    "gerav___llama-3.2-3b-instruct___reddit_in_domain": (GerAV, [], {"base_path": "./lora_configs/reddit_in_domain", "tuned_model": "llama-3.2-3b-instruct", "tune_dataset": "reddit_in_domain", "seed": 42, "custom_threshold": None}),
    "gerav___llama-3.2-3b-instruct___reddit_cross_domain": (GerAV, [], {"base_path": "./lora_configs/reddit_cross_domain", "tuned_model": "llama-3.2-3b-instruct", "tune_dataset": "reddit_cross_domain", "seed": 42, "custom_threshold": None}),
    "gerav___llama-3.2-3b-instruct___reddit_profile_based": (GerAV, [], {"base_path": "./lora_configs/reddit_profile", "tuned_model": "llama-3.2-3b-instruct", "tune_dataset": "reddit_profile_based", "seed": 42, "custom_threshold": None}),
    "gerav___llama-3.2-3b-instruct___mixed": (GerAV, [], {"base_path": "./lora_configs/mix", "tuned_model": "llama-3.2-3b-instruct", "tune_dataset": "mix_reddit_twitter", "seed": 42, "custom_threshold": None}),

    "gerav___llama-3.1-8b-instruct___twitter": (GerAV, [], {"base_path": "./lora_configs/rerun_configs_twitter", "tuned_model": "llama-3.1-8b-instruct", "tune_dataset": "twitter", "seed": 42, "custom_threshold": None}),
    "gerav___llama-3.1-8b-instruct___reddit_in_domain": (GerAV, [], {"base_path": "./lora_configs/configs_reddit_in_domain", "tuned_model": "llama-3.1-8b-instruct", "tune_dataset": "reddit_in_domain", "seed": 42, "custom_threshold": None}),
    "gerav___llama-3.1-8b-instruct___reddit_cross_domain": (GerAV, [], {"base_path": "./lora_configs/configs_reddit_cross_domain", "tuned_model": "llama-3.1-8b-instruct", "tune_dataset": "reddit_cross_domain", "seed": 42, "custom_threshold": None}),
    "gerav___llama-3.1-8b-instruct___reddit_profile_based": (GerAV, [], {"base_path": "./lora_configs/configs_reddit_profile", "tuned_model": "llama-3.1-8b-instruct", "tune_dataset": "reddit_profile_based", "seed": 42, "custom_threshold": None}),
    "gerav___llama-3.1-8b-instruct___mixed": (GerAV, [], {"base_path": "./lora_configs/configs_mix", "tuned_model": "llama-3.1-8b-instruct", "tune_dataset": "mix_reddit_twitter", "seed": 42, "debug": False, "custom_threshold": None}),

    "gerav___qwen-2.5-7b-instruct___twitter": (GerAV, [], {"base_path": "./lora_configs/rerun_configs_twitter", "tuned_model": "qwen-2.5-7b-instruct", "tune_dataset": "twitter", "seed": 42, "custom_threshold": None}),
    "gerav___qwen-2.5-7b-instruct___reddit_in_domain": (GerAV, [], {"base_path": "./lora_configs/configs_reddit_in_domain", "tuned_model": "qwen-2.5-7b-instruct", "tune_dataset": "reddit_in_domain", "seed": 42, "custom_threshold": None}),
    "gerav___qwen-2.5-7b-instruct___reddit_cross_domain": (GerAV, [], {"base_path": "./lora_configs/configs_reddit_cross_domain", "tuned_model": "qwen-2.5-7b-instruct", "tune_dataset": "reddit_cross_domain", "seed": 42, "custom_threshold": None}),
    "gerav___qwen-2.5-7b-instruct___reddit_profile_based": (GerAV, [], {"base_path": "./lora_configs/configs_reddit_profile", "tuned_model": "qwen-2.5-7b-instruct", "tune_dataset": "reddit_profile_based", "seed": 42, "custom_threshold": None}),
    "gerav___qwen-2.5-7b-instruct___mixed": (GerAV, [], {"base_path": "./lora_configs/configs_mix", "tuned_model": "qwen-2.5-7b-instruct", "tune_dataset": "mix_reddit_twitter", "seed": 42, "custom_threshold": None}),

    "gerav___gemma-3-12b-it___twitter": (GerAV, [], {"base_path": "./lora_configs/rerun_configs_twitter", "tuned_model": "gemma-3-12b-it", "tune_dataset": "twitter", "seed": 42, "custom_threshold": None}),
    "gerav___gemma-3-12b-it___reddit_in_domain": (GerAV, [], {"base_path": "./lora_configs/configs_reddit_in_domain", "tuned_model": "gemma-3-12b-it", "tune_dataset": "reddit_in_domain", "seed": 42, "custom_threshold": None}),
    "gerav___gemma-3-12b-it___reddit_cross_domain": (GerAV, [], {"base_path": "./lora_configs/configs_reddit_cross_domain", "tuned_model": "gemma-3-12b-it", "tune_dataset": "reddit_cross_domain", "seed": 42, "custom_threshold": None}),
    "gerav___gemma-3-12b-it___reddit_profile_based": (GerAV, [], {"base_path": "./lora_configs/configs_reddit_profile", "tuned_model": "gemma-3-12b-it", "tune_dataset": "reddit_profile_based", "seed": 42, "custom_threshold": None}),
    "gerav___gemma-3-12b-it___mixed": (GerAV, [], {"base_path": "./lora_configs/configs_mix", "tuned_model": "gemma-3-12b-it", "tune_dataset": "mix_reddit_twitter", "seed": 42, "custom_threshold": None}),

    "gerav___lammlein___twitter": (GerAV, [], {"base_path": "./lora_configs/twitter", "tuned_model":"lammlein", "tune_dataset": "twitter", "seed": 42, "custom_threshold": 0, "stop_tokens": [" Ja", " Nein", " ja", " nein", "JA", "NEIN"], "positive_token": "Ja", "negative_token": "Nein"}),
    "gerav___lammlein___reddit_in_domain": (GerAV, [], {"base_path": "./lora_configs/configs_reddit_in_domain", "tuned_model":"lammlein", "tune_dataset": "reddit_in_domain", "seed": 42, "custom_threshold": 0, "stop_tokens": [" Ja", " Nein", " ja", " nein", "JA", "NEIN"], "positive_token": "Ja", "negative_token": "Nein"}),
    "gerav___lammlein___reddit_cross_domain": (GerAV, [], {"base_path": "./lora_configs/configs_reddit_cross_domain", "tuned_model":"lammlein", "tune_dataset": "reddit_cross_domain", "seed": 42, "custom_threshold": 0, "stop_tokens": [" Ja", " Nein", " ja", " nein", "JA", "NEIN"], "positive_token": "Ja", "negative_token": "Nein"}),
    "gerav___lammlein___reddit_profile_based": (GerAV, [], {"base_path": "./lora_configs/configs_reddit_profile", "tuned_model":"lammlein", "tune_dataset": "reddit_profile_based", "seed": 42, "custom_threshold": 0, "stop_tokens": [" Ja", " Nein", " ja", " nein", "JA", "NEIN"], "positive_token": "Ja", "negative_token": "Nein"}),
    "gerav___lammlein___mixed": (GerAV, [], {"base_path": "./lora_configs/configs_mix", "tuned_model":"lammlein", "tune_dataset": "mix_reddit_twitter", "seed": 42, "custom_threshold": 0, "stop_tokens": [" Ja", " Nein", " ja", " nein", "JA", "NEIN"], "positive_token": "Ja", "negative_token": "Nein"}),
}

DATASET_DICT = {
    "mixed": "your dataset here",
    "twitter": "your dataset here",
    "reddit_in_domain": "your dataset here",
    "reddit_cross_domain": "your dataset here",
    "reddit_profile_based": "your dataset here",
}

class MetricEvaluator:
    def __init__(self, model_list=None, dataset_list=None):
        if model_list == None:
           self.model_list = MODEL_DICT.keys()
        else:
            self.model_list = model_list
        if dataset_list == None:
            self.dataset_list = DATASET_DICT.keys()
        else:
            self.dataset_list = dataset_list


    def save_load(self, dataset):
        try:
            ds = load_from_disk(dataset)
        except:
            ds = load_dataset(dataset)
        return ds

    def evaluate(self, num_samples=500, seed=42, output_dir="thresholds"):
        if num_samples > 0:
            datasets = [self.save_load(DATASET_DICT[ds_name])['validation'].select(range(num_samples)) for ds_name in
                        self.dataset_list]
        else:
            datasets = [self.save_load(DATASET_DICT[ds_name])['validation'] for ds_name in self.dataset_list]

        for model_name in self.model_list:
            import os
            print(f"Evaluating model: {model_name}")
            try:
                model = MODEL_DICT[model_name][0](*MODEL_DICT[model_name][1], **MODEL_DICT[model_name][2])
            except Exception as e:
                print(f"Error evaluating model {model_name}: \n{e}")
                continue
            for ds, dataset_name in zip(datasets, self.dataset_list):
                labels = [d["label"] for d in ds]
                preds, probs, generated_text = model([d["post_a"]["text"] for d in ds], [d["post_b"]["text"] for d in ds])
                print(np.min(probs), np.max(probs))
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, probs)
                specifity = 1-fpr
                youden_j = tpr + specifity - 1
                best_idx = np.argmax(youden_j)
                best_threshold = thresholds[best_idx]
                best_j = youden_j[best_idx]
            
            # Clean up model and free vram
            del model
            gc.collect()
            torch.cuda.empty_cache()

            print(f"Best threshold: {best_threshold}")


            output_path = f"{output_dir}/best-threshold_{model_name}.txt"
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(f"{best_threshold}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AV baselines")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="outputs/thresholds", help="Directory to save the results")
    parser.add_argument("--model_list", type=str, nargs='*', default=None, help="List of models to evaluate")
    parser.add_argument("--dataset_list", type=str, nargs='*', default=None, help="List of datasets to evaluate")
    
    args = parser.parse_args()

    evaluator = MetricEvaluator(model_list=args.model_list, dataset_list=args.dataset_list)
    evaluator.evaluate(num_samples=args.num_samples, output_dir=args.output_dir)


import os
import sys
import argparse
import datetime
import gc
import torch
from tqdm import tqdm
#from datasets import load_from_disk, load_dataset
import pandas as pd
import json
import numpy as np
import sklearn
from sklearn.metrics import roc_curve

#from av_baselines.GerAV import GerAV
#from av_baselines.MStyleDistance import MStyleDistance
#from FeatureDifferenceEval import FeatureDifference
#from PPMEval import PPM
#from XGBoost import XGBOOST
from SbertEval import SBERT
#from MSR import MSR
#from AdhominemApply2 import ADHOMINEM
#from valla.methods.FeatureDifferenceGerman import (vectorize, CustomTfIdfTransformer)
#from valla.methods.PPM_AV import (eval_sample, distance)
#from valla.methods.torched_AdHominem import (AdHominem, AVDataset as BaseAVDataset, load_model_from_disk, evaluate_model, modified_contrastive_loss)



'''    "m_style_distance_twitter": (MStyleDistance, [], {"threshold": "twitter"}),
    "m_style_distance_reddit": (MStyleDistance, [], {"threshold": "reddit"}),
    "gerav__gemma_3_1b_it__twitter": (GerAV, [], {"tuned_model": "gemma-3-1b-it", "tune_dataset": "twitter", "seed": 42}),
    "gerav__gemma_3_4b_it__twitter": (GerAV, [], {"tuned_model": "gemma-3-4b-it", "tune_dataset": "twitter", "seed": 42}),
    #"gerav__gemma_3_12b_it__twitter": (GerAV, [], {"tuned_model": "gemma-3-12b-it", "tune_dataset": "twitter", "seed": 42}),
    "gerav__llama-3.1-8b-instruct__twitter": (GerAV, [], {"tuned_model": "llama-3.1-8b-instruct", "tune_dataset": "twitter", "seed": 42}),
    "gerav__llama-3.2-1b-instruct__twitter": (GerAV, [], {"tuned_model": "llama-3.2-1b-instruct", "tune_dataset": "twitter", "seed": 42}),
    "gerav__llama-3.2-3b-instruct__twitter": (GerAV, [], {"tuned_model": "llama-3.2-3b-instruct", "tune_dataset": "twitter", "seed": 42}),
    "gerav__qwen-2.5-3b-instruct__twitter": (GerAV, [], {"tuned_model": "qwen-2.5-3b-instruct", "tune_dataset": "twitter", "seed": 42}),
    "gerav__qwen-2.5-7b-instruct__twitter": (GerAV, [], {"tuned_model": "qwen-2.5-7b-instruct", "tune_dataset": "twitter", "seed": 42}),
    "gemma_3_1b_it": (GerAV, [], {"tuned_model": "gemma-3-1b-it", "tune_dataset": "twitter", "seed": 42, "baseline": True}),
    "gemma_3_4b_it": (GerAV, [], {"tuned_model": "gemma-3-4b-it", "tune_dataset": "twitter", "seed": 42, "baseline": True}),
    "gemma_3_12b_it": (GerAV, [], {"tuned_model": "gemma-3-12b-it", "tune_dataset": "twitter", "seed": 42, "baseline": True}),
    "llama-3.1-8b-instruct": (GerAV, [], {"tuned_model": "llama-3.1-8b-instruct", "tune_dataset": "twitter", "seed": 42, "baseline": True}),
    "llama-3.2-1b-instruct": (GerAV, [], {"tuned_model": "llama-3.2-1b-instruct", "tune_dataset": "twitter", "seed": 42, "baseline": True}),
    "llama-3.2-3b-instruct": (GerAV, [], {"tuned_model": "llama-3.2-3b-instruct", "tune_dataset": "twitter", "seed": 42, "baseline": True}),
    "qwen-2.5-3b-instruct": (GerAV, [], {"tuned_model": "qwen-2.5-3b-instruct", "tune_dataset": "twitter", "seed": 42, "baseline": True}),
    "qwen-2.5-7b-instruct": (GerAV, [], {"tuned_model": "qwen-2.5-7b-instruct", "tune_dataset": "twitter", "seed": 42, "baseline": True}),
MODEL_DICT =  {
    "ngram_twitter": (FeatureDifference, [], {"model_type": "twitter"}),
    "ppm_twitter": (PPM, [], {"model_type": "twitter"}),
    "sbert_twitter": (SBERT, [], {"model_type": "twitter"}),
    "xgb_twitter": (XGBOOST, [], {"model_type": "twitter"}),
    "adhominem_twitter": (ADHOMINEM, [], {"model_type": "twitter"})'''
MODEL_DICT =  {
    "sbert_storyforum": (SBERT, [], {"model_type": "story_forum"}),
}

DATASET_DICT = {
    "twitter": "nllg/twitter_final",
    "reddit": "nllg/reddit_domain_datasets",
    "reddit_profile": "nllg/reddit_profile_based",
    #"twitter_old" : "nllg/twitter-av-de-2019"
    #"twitter": "nllg/twitter_revised",
    #"reddit": "/ceph/cleiter/ALIAS/reddit-av-preprocessing/data/processed/hf_dataset"
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

    def compute_stats(self, labels, preds, probs):
        '''
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * tp / (2 * tp + fp + fn)
        '''
        accuracy = sklearn.metrics.accuracy_score(labels, preds)
        precision = sklearn.metrics.precision_score(labels, preds)
        recall = sklearn.metrics.recall_score(labels, preds)
        f1_score = sklearn.metrics.f1_score(labels, preds)
        roc_auc = sklearn.metrics.roc_auc_score(labels, probs)
        # AUC Metric
        # Evtl. Pan Workshop metrics
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "roc_auc": roc_auc
        }

    def compute_stats_org(self, tp, fp, tn, fn):
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * tp / (2 * tp + fp + fn)
        # AUC Metric
        # Evtl. Pan Workshop metrics
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    def save_load(self, dataset):
        try:
            ds = load_from_disk(dataset)
        except:
            ds = load_dataset(dataset)
        return ds

    def load_jsonl_dataset(self, file_path):
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        return data

    def evaluate(self, num_samples=500, seed=42, output_dir="thresholds"):
        datasets = [self.save_load(DATASET_DICT[ds_name])['test'].shuffle(seed=seed).select(range(num_samples)) for ds_name in self.dataset_list]
        self.dataset_list = ["story_forum"]
        #for ds, ds_name in zip(datasets, self.dataset_list):
        #    print(f"Dataset: {ds_name}")
        #    for i in range(10):
        #        print(ds[i])
 
        
        for model_name in self.model_list:
            # ignore model if file exists
            import os
            #if os.path.exists(f"{output_dir}/av_baseline_results_{model_name}_{num_samples}.tsv"):
            #    print(f"Skipping model {model_name} as results already exist.")
            #    continue
            all_results = []
            print(f"Evaluating model: {model_name}")
            model = MODEL_DICT[model_name][0](*MODEL_DICT[model_name][1], **MODEL_DICT[model_name][2])
            for ds, dataset_name in zip(datasets, self.dataset_list):
                labels = [d["label"] for d in ds]
                if "xg" in model_name:
                    probas, preds = model.__call__([d["post_a"]["text"] for d in ds], [d["post_b"]["text"] for d in ds], dataset_name, labels)
                else:
                    probas, preds = model.__call__([d["post_a"]["text"] for d in ds], [d["post_b"]["text"] for d in ds])
                print(np.min(probas), np.max(probas))
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, probas)
                specifity = 1-fpr
                youden_j = tpr + specifity - 1
                best_idx = np.argmax(youden_j)
                best_threshold = thresholds[best_idx]
                best_j = youden_j[best_idx]
            
            # Clean up model and free vram
            del model
            gc.collect()
            torch.cuda.empty_cache()

            print(f"Best threshold: {best_threshold} with {best_j}")


            output_path = f"{output_dir}/best-threshold_{model_name}.txt"
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(f"{best_threshold}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AV baselines")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="thresholds/", help="Directory to save the results")
    parser.add_argument("--model_list", type=str, nargs='*', default=None, help="List of models to evaluate")
    parser.add_argument("--dataset_list", type=str, nargs='*', default=None, help="List of datasets to evaluate")
    
    args = parser.parse_args()

    evaluator = MetricEvaluator(model_list=args.model_list, dataset_list=args.dataset_list)
    evaluator.evaluate(num_samples=args.num_samples, output_dir=args.output_dir)


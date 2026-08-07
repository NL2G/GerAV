import os
import pickle
import logging
import csv
from multiprocessing import Pool
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from valla.dsets.loaders import get_av_dataset, get_av_dataset_json
from valla.utils.eval_metrics import av_metrics
from valla.utils.eval_metrics import binarize
from valla.utils.eval_metrics import evaluate_all
from valla.methods.PPM_AV import eval_sample
from valla.methods.PPM_AV import distance
from valla.utils.eval_metrics import threshold_search

class PPM:
    def __init__(self, model_type, ppm_order=5, num_workers=4):
        #add paths to ppm model folders here
        self.twitter_model = ""
        self.cross_model = ""
        self.in_model = ""
        self.profile_model = ""
        self.mixed_model = ""
        if model_type == "twitter":
            self.model_path = self.twitter_model
        if model_type == "cross":
            self.model_path = self.cross_model
        if model_type == "in":
            self.model_path = self.in_model
        if model_type == "profile":
            self.model_path = self.profile_model
        if model_type == "mixed":
            self.model_path = self.mixed_model
        with open(os.path.join(self.model_path, "logreg_ppm_5_1.clf"), 'rb') as f:
            self.logreg = pickle.load(f)
        self.num_workers = num_workers
        self.ppm_order = ppm_order

    def __call__(self, i1, i2, threshold=0.5):
        if not isinstance(i1, list):
            i1, i2, = [i1], [i2]
        probas_and_labels = []
        async_results = {}
        idx = 0
        with Pool(processes = self.num_workers) as pool:
            for txt0, txt1 in zip (i1, i2):
                async_results[idx] = pool.apply_async(
                    eval_sample, (txt0, txt1, self.ppm_order, self.logreg, None)
                )
                idx += 1
            done = False
            while not done:
                remove_idxs = []
                for idx, result in async_results.items():
                    if result.ready():
                        probas_and_labels.append(result.get())
                        remove_idxs.append(idx)
                for idx in remove_idxs:
                    del async_results[idx]
                if len(async_results) == 0:
                    done = True
        #print(probas_and_labels)
        probas = [p for p in probas_and_labels]
        preds = (np.array(probas) >= threshold).astype(int)
        return probas, preds.tolist()

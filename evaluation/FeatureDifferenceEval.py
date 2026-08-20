import os
import csv
import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc
import argparse
from valla.methods.FeatureDifferenceGerman import vectorize
from valla.methods.FeatureDifferenceGerman import CustomTfIdfTransformer
from valla.utils.eval_metrics import av_metrics
from valla.utils.eval_metrics import evaluate_all
from valla.utils.eval_metrics import threshold_search

class FeatureDifference:
    def __init__(self, model_type):
        #add paths to your ngram model folders here
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
        with open(os.path.join(self.model_path, "large_model.p"), 'rb') as f:
            self.clf, self.transformer, self.scaler, self.secondary_scaler = pickle.load(f)
        self.vectorize = vectorize

    def __call__(self, i1, i2, threshold=0.5):
        if not isinstance(i1, list):
            i1, i2, = [i1], [i2]


        num_samples = len(i1)
        feature_sz = len(self.transformer.get_feature_names_out())
        XX_test = np.zeros((num_samples, feature_sz), dtype=np.float32)
        Y_dummy = np.zeros(num_samples, dtype=np.float32)
        test_idxs = np.arange(num_samples)
        #test_idxs = np.array(range(test_sz))
        #np.random.shuffle(test_idxs)
        input_list= [(0, a, b) for a, b in zip(i1, i2)]

        vectorize(
            XX_test,
            Y_dummy,
            test_idxs,
            self.transformer,
            self.scaler,
            self.secondary_scaler,
            input_list,
            num_samples,
            eval=True
        )
        probs = self.clf.predict_proba(XX_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        return preds.tolist(), probs, None
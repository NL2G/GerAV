import os
import pickle
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses.ContrastiveLoss import SiameseDistanceMetric
import torch

def contrastive_predict(model, i1, i2, kernel_fn, threshold=0.5, distance_metric=SiameseDistanceMetric.EUCLIDEAN):
    if not isinstance(i1, list):
        i1, i2 = [i1], [i2]

    model.eval()
    with torch.no_grad():
        emb1 = model.encode(i1, show_progress_bar=False)
        emb2 = model.encode(i2, show_progress_bar=False)

        distances = distance_metric(torch.tensor(emb1), torch.tensor(emb2)).cpu().numpy()
        similarities = kernel_fn(torch.tensor(distances)).cpu().numpy()
        preds = (similarities >= threshold).astype(int)

    #model.train()
    return similarities, preds.tolist()


class SBERT:
    def __init__(self, model_type):
        # add paths to sber model folders here
        self.twitter_model = ""
        self.cross_model = ""
        self.in_model = ""
        self.profile_model = ""
        self.mixed_model = ""

        if model_type == "twitter":
            self.model_path = self.twitter_model
        elif model_type == "cross":
            self.model_path = self.cross_model
        elif model_type == "in":
            self.model_path = self.in_model
        elif model_type == "profile":
            self.model_path = self.profile_model
        elif model_type == "mixed":
            self.model_path = self.mixed_model
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Load SentenceTransformer model
        self.model = SentenceTransformer(self.model_path)

        self.distance_metric = SiameseDistanceMetric.EUCLIDEAN
        self.kernel_fn = lambda x: 1 / (1 + x)  # matches training kernel

    def __call__(self, i1, i2, threshold=0.5):
        # Return 0/1 predictions
        similarities, preds = contrastive_predict(
            model=self.model,
            i1=i1,
            i2=i2,
            kernel_fn=self.kernel_fn,
            threshold=threshold,
            distance_metric=self.distance_metric
        )
        return similarities, preds
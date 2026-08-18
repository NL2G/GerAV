from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import numpy as np


class MStyleDistance:
    def __init__(self, threshold=None):
        self.model = SentenceTransformer('StyleDistance/mstyledistance')
        self.threshold = threshold

    def __call__(self, i1, i2):
        if isinstance(i1, list):
            sim = [self.compute_cos_sim(s1, s2) for s1, s2 in zip(i1, i2)]
        else:
            sim = self.compute_cos_sim(i1, i2)
        
        if self.threshold is None:
            return sim  
        
        labels = [s >= self.threshold for s in sim]
        return labels, sim, None

    def compute_cos_sim(self, i1, i2):
        input_embedding = self.model.encode(i1)
        other_embeddings = self.model.encode(i2)
        sim = cos_sim(input_embedding, other_embeddings)
        return sim.item()



if __name__ == "__main__":
    msd = MStyleDistance(threshold=0)

    i1 = ["Hi, ich bin Autor A.", "Some text from author A."]
    i2 = ["Hi, ich bin Autor A.", "Some text from author B."]
    pred, prob, generations = msd(i1, i2)
    
    print("Predictions:", pred)
    print("Probabilities:", prob)
    print("Generations:", generations)

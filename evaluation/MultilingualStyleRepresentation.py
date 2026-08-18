from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


class MultilingualStyleRepresentation:
    def __init__(self, threshold=None):
        self.model = SentenceTransformer('Blablablab/multilingual-style-representation')
        self.threshold = threshold

    def __call__(self, i1, i2):
        if isinstance(i1, list):
            sim = [self.compute_sim(s1, s2) for s1, s2 in zip(i1, i2)]
        else:
            sim = self.compute_sim(i1, i2)
        
        if self.threshold is None:
            return sim
        labels = [s >= self.threshold for s in sim]
        print(labels, sim)
        return labels, sim, None

    def compute_sim(self, i1, i2):
        input_embedding = self.model.encode(i1)
        other_embeddings = self.model.encode(i2)
        sim = self.model.similarity(input_embedding, other_embeddings)
        return sim.item()
    

if __name__ == "__main__":
    msr = MultilingualStyleRepresentation(threshold=0)
    i1 = ["Hi, ich bin Autor A.", "Some text from author A."]
    i2 = ["Hi, ich bin Autor A.", "Some text from author B."]
    pred, prob, generations = msr(i1, i2)

    print("Predictions:", pred)
    print("Probabilities:", prob)
    print("Generations:", generations)

    
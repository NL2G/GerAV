# first just build a generic loader for the aa and av data formats - read them in as a list of samples
import csv
import pandas as pd
from typing import List, Union
import sys
import csv
from datasets import load_from_disk

# kinda dirty but need for PAN20 - I don't really wanna chop data or something
csv.field_size_limit(sys.maxsize)


def aa_as_pandas(data: List[List[Union[int, str]]]) -> pd.DataFrame:
    return pd.DataFrame(data, columns=['labels', 'text'])


def av_as_pandas(data: List[List[Union[int, str]]]) -> pd.DataFrame:
    return pd.DataFrame(data, columns=['same/diff', 'text0', 'text1'])


def get_aa_dataset(dataset_path: str) -> List[List[Union[int, str]]]:
    data = []
    with open(dataset_path, 'r', errors='ignore') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i > 0:  # skip header
                data.append([int(line[0]), str(line[1])])
    return data


def get_av_dataset(dataset_path: str) -> List[List[Union[int, str, str]]]:
    data = []
    with open(dataset_path, 'r', errors='ignore') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i > 0:  # skip header
                data.append([int(line[0]), str(line[1]), str(line[2])])
    return data

    def get_av_dataset_gerav(
        dataset_path: str,
        split: str,
    ) -> List[List[Union[int, str, str]]]:

        dataset = load_from_disk(dataset_path)[split]

        data = []

        for obj in dataset:
            label = int(obj["label"])
            text_a = obj["post_a"]["text"]
            text_b = obj["post_b"]["text"]

            data.append([label, text_a, text_b])

        return data


def get_aa_as_pandas(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path, header=0, names=['labels', 'text'])


def get_av_as_pandas(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path, header=0, names=['same/diff', 'text0', 'text1'])

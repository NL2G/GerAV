from datasets import Dataset, load_dataset, DatasetDict, concatenate_datasets
import pandas as pd

def select_subset(dataset, count = 10000):
    first_half = dataset.select(range(0, len(dataset) // 2))
    second_half = dataset.select(range(len(dataset) // 2, len(dataset)))
    first_half = first_half.shuffle(seed=42).select(range(count))
    second_half = second_half.shuffle(seed=42).select(range(count))

    # ensure post_a["user_id"] and post_b["user_id"] are string
    ds = concatenate_datasets([first_half, second_half])
    ds = ds.map(lambda x: {"post_a": {k: str(v) for k, v in x["post_a"].items()}, "post_b": {k: str(v) for k, v in x["post_b"].items()}})
    
    return ds

# to build a mixed dataset, either include all reddit datasets + twitter here and adapt the counts below 
# or alternatively only include twitter and then concatenate it with our mixed reddit dataset (after texts are retrieved for the urls)
datasets = ["datasets to include in the mix"]
datasets_loaded = [load_dataset(ds) for ds in datasets]
train_dataframes, val_dataframes, test_dataframes = [], [], []
for i, ds in enumerate(datasets_loaded):
    ds_train = select_subset(ds["train"], count=20000).to_pandas()
    ds_val = select_subset(ds["validation"], count=4000).to_pandas()
    ds_test = select_subset(ds["test"], count=4000).to_pandas()
 
    train_dataframes.append(ds_train)
    val_dataframes.append(ds_val)
    test_dataframes.append(ds_test)

train_df = pd.concat(train_dataframes).reset_index(drop=True)
val_df = pd.concat(val_dataframes).reset_index(drop=True)
test_df = pd.concat(test_dataframes).reset_index(drop=True)


full_ds = DatasetDict(
    {
        "train": Dataset.from_pandas(train_df).shuffle(seed=42),
        "validation": Dataset.from_pandas(val_df).shuffle(seed=42),
        "test": Dataset.from_pandas(test_df).shuffle(seed=42),
    }
)



full_ds.save_to_disk("outputs/datasets/mini_mix_dataset")


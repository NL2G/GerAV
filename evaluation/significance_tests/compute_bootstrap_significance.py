import os
import glob
import json
import ast
import re
from pathlib import Path
import random
import pandas as pd
import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from argparse import ArgumentParser


N_BOOTSTRAPS = 10000
SEED = 42


def compute_f1(row):
    y_true = row["true_labels"]
    y_pred = row["predicted_labels"]
    return f1_score(y_true, y_pred, zero_division=0)


def compute_accuracy(row):
    return accuracy_score(row["true_labels"], row["predicted_labels"])


def get_bootstrap_metrics(
    pred_labels,
    choice_lists,
    true_label_preds,
):
    """
    Computes paired bootstrap distributions for both F1 and Accuracy.
    """

    pred_labels = np.asarray(pred_labels)

    f1_preds = []
    acc_preds = []

    for choice_list, true_labels_perm in zip(
        choice_lists,
        true_label_preds
    ):
        pred_perm = pred_labels[choice_list]

        f1_preds.append(
            f1_score(true_labels_perm, pred_perm)
        )

        acc_preds.append(
            accuracy_score(true_labels_perm, pred_perm)
        )

    return f1_preds, acc_preds


def main():


    parser = ArgumentParser()
    parser.add_argument("--in_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    args.out_dir.mkdir(parents=True, exist_ok=True,)
    out_dir = Path(args.out_dir)

    tsv_files = glob.glob(str(args.in_dir / "av*.tsv"))

    # Store all results
    dfs = []

    # Store bootstrap samples per dataset
    idx_choice_per_dataset = {}
    true_label_preds_per_dataset = {}


    for file in tqdm.tqdm(tsv_files):

        df = pd.read_csv(file, sep="\t",)

        columns = [
            "probabilities",
            "true_labels",
            "dataset",
            "model",
        ]

        if "threshold" in df.columns:
            columns.append("threshold")

        df = df[columns]

        df = df.rename(columns={"dataset": "evaluated_dataset"})

        if "threshold" not in df.columns:
            df["threshold"] = 0

        df["filename"] = os.path.basename(file)
        try:
            df["probabilities"] = df["probabilities"].apply(ast.literal_eval)
        except Exception:
            df["probabilities"] = df["probabilities"].apply(
                lambda x: list(map(float, re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", str(x)))))
        try:
            df["true_labels"] = df["true_labels"].apply(ast.literal_eval)
        except Exception:
            df["true_labels"] = df["true_labels"].apply(ast.literal_eval)
            
        df["predicted_labels"] = df.apply(lambda row: [1 if p >= row["threshold"] else 0 for p in row["probabilities"]], axis=1)

        df["f1"] = df.apply(compute_f1, axis=1)
        df["accuracy"] = df.apply(compute_accuracy, axis=1)

        # Get the same permutation lists for each dataset to ensure fair comparison
        for dataset in df["evaluated_dataset"].unique():
            content_length = len(df[df["evaluated_dataset"] == dataset].iloc[0]["true_labels"])
            if dataset not in idx_choice_per_dataset:
                idx_choice_per_dataset[dataset] = [np.random.choice(content_length, content_length, replace=True) for _ in range(N_BOOTSTRAPS)]
                true_label_preds_per_dataset[dataset] = [
                    np.array(df[df["evaluated_dataset"] == dataset].iloc[0]["true_labels"])[choice_list] for choice_list
                    in idx_choice_per_dataset[dataset]]

        bootstrap_metrics = df.apply(
            lambda row: get_bootstrap_metrics(
                row["predicted_labels"],
                idx_choice_per_dataset[row["evaluated_dataset"]],
                true_label_preds_per_dataset[row["evaluated_dataset"]],
            ),
            axis=1,
        )

        df["f1_preds"] = bootstrap_metrics.apply(lambda x: x[0])
        df["accuracy_preds"] = bootstrap_metrics.apply(lambda x: x[1])
        dfs.append(df)

    # Save idx choice to json
    with open(out_dir / "idx_choice_per_dataset.json", "w") as f:
        json.dump({dataset: [choice_list.tolist() for choice_list in choice_lists] for dataset, choice_lists in
                   idx_choice_per_dataset.items()}, f)

    with open(out_dir / "true_label_per_per_dataset.json", "w") as f:
        json.dump({dataset: [true_labels_perm.tolist() for true_labels_perm in true_labels_perms] for
                   dataset, true_labels_perms in true_label_preds_per_dataset.items()}, f)

    df = pd.concat(dfs, ignore_index=True)
    # sort by f1_score
    df = df.sort_values(by="f1", ascending=False)

    df.to_csv(out_dir / "bootstrap_results.csv", index=False)

    print(
        f"Finished. Saved results to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
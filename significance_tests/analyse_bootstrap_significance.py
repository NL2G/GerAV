import ast
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import tqdm
import numpy as np


def compute_adjacent_ranking(df_ds, metric, preds_col):
    """
    Rank models by a metric and compare adjacent models using paired bootstrap.
    """

    # Ranking models
    df_ds = df_ds.sort_values(by=metric, ascending=False).reset_index(drop=True)

    ranked_models = df_ds["model"].tolist()

    perm_lookup = {
        row["model"]: ast.literal_eval(row[preds_col])
        for _, row in df_ds.iterrows()
    }

    results = []

    for i in range(len(ranked_models) - 1):
        model_a = ranked_models[i]
        model_b = ranked_models[i + 1]

        score_a = df_ds[df_ds["model"] == model_a][metric].values[0]
        score_b = df_ds[df_ds["model"] == model_b][metric].values[0]

        scores_a = perm_lookup[model_a]
        scores_b = perm_lookup[model_b]

        diffs = np.asarray(scores_a) - np.asarray(scores_b)

        mean_diff = np.mean(diffs)

        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

        p = 2 * min(
            np.mean(diffs <= 0),
            np.mean(diffs >= 0)
        )

        significance = ci_low > 0 or ci_high < 0

        results.append({
            "model:a": model_a,
            "model:b": model_b,
            "rank_a": i + 1,
            "rank_b": i + 2,
            f"{metric}_a": score_a,
            f"{metric}_b": score_b,
            "mean_diff": mean_diff,
            "p_value": p,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "significance": significance,
        })

    return results


def main():

    parser = ArgumentParser()
    parser.add_argument("--in_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True,)

    df = pd.read_csv(args.in_file)

    for dataset in tqdm.tqdm(df["evaluated_dataset"].unique()):

        df_ds = df[df["evaluated_dataset"] == dataset].copy()

        # F1 ranking
        f1_results = compute_adjacent_ranking(df_ds, metric="f1", preds_col="f1_preds")

        for r in f1_results:
            r["dataset"] = dataset
            r["metric"] = "f1"


        # Accuracy ranking
        accuracy_results = compute_adjacent_ranking(df_ds, metric="accuracy", preds_col="accuracy_preds")

        for r in accuracy_results:
            r["dataset"] = dataset
            r["metric"] = "accuracy"


        results = f1_results + accuracy_results

        out_path = (args.out_dir / f"{dataset}_ranked_results.csv")

        pd.DataFrame(results).to_csv(out_path, index=False)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
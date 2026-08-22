import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_NGRAM = os.path.join(BASE_DIR, "FeatureDifference_GerAV.py")
TRAIN_PPM = os.path.join(BASE_DIR, "PPM_AV_GerAV.py")
TRAIN_ADH = os.path.join(BASE_DIR, "torched_AdHominem_GerAV.py")
TRAIN_SBERT = os.path.join(BASE_DIR, "SiameseBert_GerAV.py")
OUT = os.path.join(BASE_DIR, "..", "results")


def disable_wandb(env):
    env["WANDB_MODE"] = "disabled"
    env["WANDB_SILENT"] = "true"
    env["WANDB_CONSOLE"] = "off"
    env["WANDB_DISABLED"] = "true"

def run_training(script_path, train_path, test_path, out_path, mode):
    out_path = out_path + os.sep
    env = os.environ.copy()
    disable_wandb(env)
    env["PYTHONPATH"] = "/GerAV/baselines/Valla"

    if mode=="ngram":
        cmd = [
            "python",
            script_path,
            "--train_path", train_path,
            "--test_path", test_path,
            "--out_path", out_path
        ]
    elif mode=="ppm":
        cmd = [
            "python",
            script_path,
            "--train_path", train_path,
            "--test_path", test_path,
            "--cache_path", out_path,
            "--dset_name", "ppm"
        ]
    elif mode=="adh":
        cmd = [
            "python",
            script_path,
            "--train_path", train_path,
            "--test_path", test_path,
            "--save_model_dir", out_path,
            "--dont_use_fasttext"
        ]
    elif mode=="sbert":
        cmd = [
            "python",
            script_path,
            "--train_path", train_path,
            "--test_path", test_path,
            "--output_path", out_path,
        ]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"Running: {' '.join(cmd)} (PYTHONPATH={env['PYTHONPATH']})")
    try:
        result = subprocess.run(
            cmd,
            text=True,
            check=True,
            env=env
        )
    except subprocess.CalledProcessError as e:
        print("Error")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to the dataset")
    parser.add_argument("--new_root", required=True, help="Name of the dataset, e.g. twitter, profile-based")
    parser.add_argument("--models", nargs="+", choices=["ppm", "ngram", "adh", "sbert"], required=True, help="Models to run")
    args = parser.parse_args()

    data_path = args.data_path
    new_root = args.new_root

    model_dirs = {
        "ngram": os.path.join(OUT, new_root, "ngram"),
        "ppm": os.path.join(OUT, new_root, "ppm"),
        "adh": os.path.join(OUT, new_root, "adh"),
        "sbert": os.path.join(OUT, new_root, "sbert"),
    }

    model_scripts = {
        "ngram": TRAIN_NGRAM,
        "ppm": TRAIN_PPM,
        "adh": TRAIN_ADH,
        "sbert": TRAIN_SBERT,
    }

    for model in args.models:
        out_dir = model_dirs[model]
        os.makedirs(out_dir, exist_ok=True)

        run_training(
            model_scripts[model],
            data_path,
            out_dir,
            model
        )

if __name__ == "__main__":
    main()
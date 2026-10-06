# GerAV <br><sub><sup>Towards New Heights in German Authorship Verification using Fine-Tuned LLMs on a New Benchmark</sup></sub>

[![📄 arXiv](https://img.shields.io/badge/View%20on%20arXiv-B31B1B?logo=arxiv&labelColor=gray)]([https://arxiv.org/abs/your-arxiv-i](https://arxiv.org/abs/2601.13711)d)






## Repository structure

- `training/`: fine-tuning pipeline and dataset loaders
- `baselines/`: code for running Valla baseline models
- `lora_configs/`: model- and dataset-specific training configs
- `paper_checkpoints/`: saved checkpoints and tuned models
- `evaluation/`: threshold tuning and evaluation code
- `data/`: data preprocessing and dataset construction scripts
- `slurm_training.sh` and `slurm_eval.sh`: Slurm job entry points

## Usage

### Dataset Creation

#### Twitter

Get access to the original dataset from:
https://zenodo.org/records/7528718
and store the files locally.

Our Twitter dataset can be reconstructed from tweet IDs provided under https://huggingface.co/datasets/nllg/GerAV-Twitter. Get access and store the ID dataset locally. 

For reconstruction of the final dataset run 
```bash
python data/preprocessing/twitter/id_to_tweet.py \
    --original_dir "path/to/original/twitter/dataset" \
    --dataset_dir "path/to/id/dataset" \
    --output_dir "path/to/output/dir"
```

#### Reddit

Our Reddit dataset can be reconstructed from the URLs provided under https://huggingface.co/datasets/nllg/GerAV-Reddit. Reddit posts can for example be accessed via the Reddit API (recommended) or with a crawler. An example crawler is provided under `data/tools/reddit_crawl_example.py`; when using it make sure to consider rate limits and other constraints. 

#### Mixed
The mixed dataset can be reconstructed from the mixed dataset permalinks provided in the GerAV-Reddit dataset, combined with a stratified 20,000/4,000/4,000 train/validation/test split of the Twitter dataset.
To create such a split, you can use `data/preprocessing/create_mix.py`.

### Training

#### Lora-Tuning

```bash
python training/train.py --config lora_configs/configs_reddit_in_domain/qwen-2.5-7b-instruct.toml
```

For cluster training:

```bash
sbatch slurm_training.sh
```

#### Valla Baselines

This part of the code is based on and adapted to our GerAV benchmark from the repository belonging to Tyo et al. (2022): https://github.com/JacobTyo/Valla.git. Please refer to the required setup and environment configuration as described in the original repository before running the baslines. 

To run a baseline:
```bash
python baslines/Valla/run_baselines.py \
    --data_path "path/to/your/dataset" \
    --new_root "dataset_name" \
    --models ngram ppm adh sbert
```
specifying which models to train on which dataset. 

### Evaluation

#### Threshold Tuning
To tune thresholds on the validation set run after adding your dataset and model paths:
```bash
HF_HOME=${HF_HOME} python av_baselines/threshold_tuning.py \
  --model_list sbert_twitter\
  --dataset_list twitter \
  --output_dir outputs/thresholds
```

#### Performance
To get scores on the test set run after adding your dataset and model paths:
```bash
HF_HOME=${HF_HOME} python av_baselines/evaluate.py \
  --model_list baseline_qwen-2.5-7b-instruct \
  --dataset_list twitter \
  --output_dir outputs/scores
```

The model and dataset names used by evaluation are defined in `av_baselines/evaluate.py`.

#### Significance
To generate bootstrap results run
```bash
python evaluation/significance_tests/compute_bootstrap_significance.py \
  --in_dir "path/to/evaluation/tsv/files" \
  --out_dir "path/to/save/bootstrap"
```
inputting the directory containing the .tsv files with performance scores from the step above and the output directory to save the resulting bootstrap .csv file.

To compare models run
```bash
python evaluation/significance_tests/anaylse_bootstrap_significance.py \
  --in_file "path/to/bootstrap_results.csv" \
  --out_dir "path/to/save/rankings"
```


For each dataset, this ranks models by F1 and accuracy and compares adjacent models using the paired bootstrap distributions.
> Notes on the datasets will follow soon

## 📖 Citation

If you use this work in your research, please cite it as:

```bibtex
@inproceedings{kiefer-etal-2026-gerav,
    title = "{G}er{AV}: Towards New Heights in {G}erman Authorship Verification using Fine-Tuned {LLM}s on a New Benchmark",
    author = "Kiefer, Lotta  and
      Leiter, Christoph  and
      Takeshita, Sotaro  and
      Schmidt, Elena  and
      Eger, Steffen",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1991/",
    doi = "10.18653/v1/2026.findings-acl.1991",
    pages = "40050--40069",
    ISBN = "979-8-89176-395-1"
}
```

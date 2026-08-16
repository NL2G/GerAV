# GerAV <br><sub><sup>Towards New Heights in German Authorship Verification using Fine-Tuned LLMs on a New Benchmark</sup></sub>

[![📄 arXiv](https://img.shields.io/badge/View%20on%20arXiv-B31B1B?logo=arxiv&labelColor=gray)]([https://arxiv.org/abs/your-arxiv-i](https://arxiv.org/abs/2601.13711)d)






## Repository structure

- `training/`: fine-tuning pipeline and dataset loaders
- `lora_configs/`: model- and dataset-specific training configs
- `paper_checkpoints/`: saved checkpoints and tuned models
- `av_baselines/`: baseline authorship verification evaluation code
- `evaluation/`: metric and threshold utilities
- `data/`: data preprocessing and dataset construction scripts
- `slurm_training.sh` and `slurm_eval.sh`: Slurm job entry points

## Usage

### Training

```bash
python training/train.py --config lora_configs/configs_reddit_in_domain/qwen-2.5-7b-instruct.toml
```

For cluster training:

```bash
sbatch slurm_training.sh
```

### Evaluation

```bash
HF_HOME=${HF_HOME} python av_baselines/apply_all_baselines.py \
  --model_list baseline_qwen-2.5-7b-instruct \
  --dataset_list twitter \
  --output_dir outputs/scores
```

The model and dataset names used by evaluation are defined in `av_baselines/apply_all_baselines.py`.

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
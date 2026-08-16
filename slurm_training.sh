#!/bin/bash
#SBATCH --job-name=sft_%a
#SBATCH --output=logs/sft_%a.out
#SBATCH --error=logs/sft_%a.err

#ADD THE PARAMETERS SPECIFIC TO YOUR CLUSTER 

#SBATCH --time=10:00:00
#SBATCH --array=0-1%2

set -euo pipefail
set -x
mkdir -p logs

###

# Load the necessary modules and virtual environment here

###

export HF_HOME=HF_HOME
export HF_TOKEN=HF_TOKEN

NUM_GPUS=2 # Should be equal to the number of GPUs allocated by SLURM

configs=(
    "./lora_configs/configs_reddit_in_domain/qwen-2.5-7b-instruct.toml"
    "./lora_configs/configs_reddit_cross_domain/qwen-2.5-7b-instruct.toml"
    # ...
)

for config in "${configs[@]}"; do
    if [ ! -f "$config" ]; then
        echo "Config file $config does not exist."
        exit 1
    fi
done

echo "Job started at: $(date)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Using config: ${configs[$SLURM_ARRAY_TASK_ID]}"
echo "Running on $NUM_GPUS GPUs"

torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    training/train.py \
    --config ${configs[$SLURM_ARRAY_TASK_ID]}
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

models=(
    "gerav___qwen-2.5-7b-instruct___reddit_in_domain___20_12"
    "gerav___qwen-2.5-7b-instruct___reddit_cross_domain___20_12"
    # ...
)

echo "Job started at: $(date)"
echo "Starting training job with SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Using model file: ${models[$SLURM_ARRAY_TASK_ID]}"

HF_HOME=${HF_HOME} python ./av_baselines/apply_all_baselines.py --model_list ${models[$SLURM_ARRAY_TASK_ID]}
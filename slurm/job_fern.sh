#!/bin/bash
#SBATCH --job-name=fern
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/fern_%j.out
#SBATCH --error=logs/fern_%j.err

# NOTE: create the log directory before submitting -- Slurm will not make it:
#     mkdir -p ~/comp3710/logs
#
# Partition notes for Rangpur (check with `sinfo -s` before submitting):
#   comp3710 / a100 / a100-grind  -> the same ten a100-[0-9] nodes, often full
#   a100-test                     -> a100-a, a100-b; QOS caps wall time at 20 min
#   p100                          -> older card, idle more often, plenty for this
# Do NOT request --mem on a100-test; the node config rejects it.

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

source $HOME/miniconda3/bin/activate
conda activate torch

cd $HOME/comp3710

python render_fern.py --system barnsley-fern
python render_fern.py --system sierpinski
python box_counting.py --system all

echo "Finished $(date)"

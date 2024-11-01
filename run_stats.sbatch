#!/bin/bash
#SBATCH --job-name=stats
#SBATCH --output=./logs/stats_%j.out
#SBATCH --error=./logs/stats_%j.err
#SBATCH --partition=day
#SBATCH --ntasks=1
# #SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --nodelist=node-gpu-02
#SBATCH --account=member
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10

# Optional: Load any required modules or set up the environment
# module load some_module

source .venv/bin/activate

# python train.py --log-writer "tensorboard" --batch-size 32
python utils/mp_GetStats.py

#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gres=gpu:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --mem=12000M              # memory per node
#SBATCH --time=0-00:30            # time (DD-HH:MM)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.7 scipy-stack
module load StdEnv/2020 gcc/9.3.0
module load rdkit/2021.03.3

source ~/env/hill-climbing/bin/activate

time python climb.py

deactivate


#!/bin/bash

module --force purge
module load nixpkgs/16.09
module load gcc/7.3.0
module load rdkit/2019.03.4
module load scipy-stack/2019b

virtualenv --no-download ~/env/hill-climbing
source ~/env/hill-climbing/bin/activate

pip install --upgrade pip
pip install pytorch-lightning
pip install selfies
pip install seaborn

deactivate

echo "Environment hill-climbing created."


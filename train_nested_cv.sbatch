#!/bin/bash

#SBATCH --job-name=nested_cv_franzosa

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G

#SBATCH --time=04:00:00

#SBATCH --qos=bbdefault

#SBATCH --account=guellili-slr-automation

#SBATCH --output=/rds/projects/g/guellili-slr-automation/mehvish/thesis/logs/nested_cv_%j.out

#SBATCH --error=/rds/projects/g/guellili-slr-automation/mehvish/thesis/logs/nested_cv_%j.err



set -e

module purge

module load bluebear

module load Python/3.9.5-GCCcore-10.3.0



PROJECT_DIR="/rds/projects/g/guellili-slr-automation/mehvish/thesis"

VENV_DIR="/rds/homes/m/mxs2085/ibd_reproduction/venv"



source "${VENV_DIR}/bin/activate"



echo "Running on node: $(hostname)"

echo "Python: $(which python3)"

echo "Start time: $(date)"



python3 "${PROJECT_DIR}/scripts/train_nested_cv.py"



echo "End time: $(date)"


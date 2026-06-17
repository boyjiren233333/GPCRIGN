# GPCRIGN

GPCRIGN is a graph neural network model for GPCR ligand function classification. This repository provides the model code and a small toy example dataset for demonstrating the input format, training workflow, and prediction output.

## Repository Structure

- code/: model, graph construction, training, and prediction scripts
- scripts/: helper scripts for preparing toy datasets
- data/toy_example/: a small 5HT1A toy dataset with train, validation, and test splits

## Toy Example Dataset

The toy example contains 200 5HT1A ligand-pocket pickle files:

- 160 samples for training
- 20 samples for validation
- 20 samples for testing

Labels are stored in data/toy_example/label.csv.

Label meaning:

- 0: agonist
- 1: antagonist

Graph cache files are generated automatically during training or prediction and are not included in this repository.

## Training

Run a quick example training job:

    python code/train.py --epochs 1 --batch-size 16 --num-processes 6

The best checkpoint is saved to model_save/best_model.pth.

## Prediction

After training, run prediction on the test split:

    python code/predict.py --batch-size 16 --num-processes 6

Prediction results are saved to prediction_results/predictions.csv.

## Build a Toy Dataset from the Full Dataset

The full dataset is not included. To recreate the toy example from the original local dataset:

    python scripts/create_toy_example.py \
      --pickle-root ../full_dataset/pickle_files \
      --label-csv ../full_dataset/label.csv \
      --output-dir data/toy_example \
      --target 5HT1A \
      --overwrite

## Notes

This repository is intended as a lightweight demonstration of the GPCRIGN input format and workflow. Large datasets, trained checkpoints, generated graph caches, and training outputs are excluded from version control.

## Environment Setup

Create the conda environment:

    conda create -n gpcrign python=3.9.18 -y
    conda activate gpcrign
    pip install -r requirements.txt

For the exact installation sequence used in this project, run:

    bash setup_environment.sh

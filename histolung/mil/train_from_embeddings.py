from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
import h5py
import pandas as pd

from histolung.models.models import MILModel
from histolung.mil.mil_trainer import TileMILTrainer
from histolung.mil.data_loader import WSIDataset
from histolung.mil.utils import get_wsi_dataloaders

# Set project directories and configuration paths
project_dir = Path(__file__).resolve().parents[2]
model_dir = project_dir / "MIL/first/"
cfg_path = model_dir / "config.yml"


def load_metadata():
    """Load WSI metadata and fold information for cross-validation."""
    fold_df = pd.read_csv(project_dir / "data/interim/tcga_folds.csv")
    with open(project_dir / "data/interim/tcga_wsi_data.json") as f:
        wsi_metadata = json.load(f)
    return wsi_metadata, fold_df


def split_by_fold(wsi_metadata, fold_df):
    """Split WSI metadata by fold for k-fold cross-validation."""
    n_folds = fold_df["fold"].max() + 1
    output = [[] for _ in range(n_folds)]
    fold_mapping = dict(zip(fold_df['wsi_id'], fold_df['fold']))

    for wsi_info in wsi_metadata:
        fold = fold_mapping.get(wsi_info['wsi_id'])
        if fold is not None:
            output[fold].append(wsi_info)

    return output, n_folds


def main():
    # Configuration
    num_epochs = 10
    label_map = {"luad": 0, "lusc": 1}
    hdf5_path = project_dir / "data/interime/embeddings.h5"  # Path to HDF5 file

    # Load metadata and split into k-folds
    wsi_metadata, fold_df = load_metadata()
    wsi_metadata_by_folds, n_folds = split_by_fold(wsi_metadata, fold_df)

    # Open HDF5 file for reading embeddings
    hdf5_file = h5py.File(hdf5_path, 'r')

    # Loop over each fold
    for fold in range(n_folds):
        print(f"Processing fold {fold+1}/{n_folds}")

        # Load the model from configuration
        model = MILModel.from_config(cfg_path)

        # Get the DataLoaders for the current fold
        dataloaders = get_dataloaders_from_hdf5(hdf5_file,
                                                wsi_metadata_by_folds, fold,
                                                label_map)

        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Initialize the trainer
        trainer = TileMILTrainer(
            model,
            dataloaders,
            optimizer,
            loss_fn,
            device='cuda',
            tile_preprocess=None,  # No preprocess needed
            tile_augmentation=None,  # No augmentation needed
        )

        # Train the model

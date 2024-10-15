from pathlib import Path
import json
import logging

import torch
from torch.utils.data import DataLoader
import h5py
import pandas as pd

from histolung.models.models import MILModel
from histolung.mil.mil_trainer import EmbeddingMILTrainer
from histolung.mil.data_loader import HDF5EmbeddingDataset
from histolung.mil.utils import (get_wsi_dataloaders, get_loss_function,
                                 get_optimizer)
from histolung.utils import yaml_load

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set project directories and configuration paths
project_dir = Path(__file__).resolve().parents[2]
config_path = "/home/valentin/workspaces/histolung/models/MIL/first/config.yaml"
model_dir = Path("/home/valentin/workspaces/histolung/models/MIL/first/")


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


def get_dataloaders_from_hdf5(
    hdf5_file,
    wsi_metadata_by_folds,
    fold,
    label_map,
    batch_size=2,
):
    """
    Create training and validation DataLoaders based on the current fold.
    """
    wsi_meta_train = [
        wsi for i, wsi_fold in enumerate(wsi_metadata_by_folds) if i != fold
        for wsi in wsi_fold
    ]
    wsi_meta_val = wsi_metadata_by_folds[fold]

    train_dataset = HDF5EmbeddingDataset(
        hdf5_file=hdf5_file,
        label_map=label_map,
        wsi_ids=wsi_meta_train,
    )
    val_dataset = HDF5EmbeddingDataset(
        hdf5_file=hdf5_file,
        label_map=label_map,
        wsi_ids=wsi_meta_val,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "val": val_loader}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Configuration
    config = yaml_load(config_path)
    num_epochs = config["training"]["epochs"]
    label_map = config["data"]["label_map"]
    hdf5_path = project_dir / config["data"]["embedding_file"]

    # Load metadata and split into k-folds
    wsi_metadata, fold_df = load_metadata()
    wsi_metadata_by_folds, n_folds = split_by_fold(wsi_metadata, fold_df)

    # Open HDF5 file for reading embeddings
    hdf5_file = h5py.File(hdf5_path, 'r')

    # Loop over each fold
    for fold in range(n_folds):
        logging.info(f"Processing fold {fold+1}/{n_folds}")

        # Load the model from configuration
        model = MILModel.from_config(config_path)

        # Get the DataLoaders for the current fold

        wsi_dataloaders = get_wsi_dataloaders(
            wsi_metadata_by_folds,
            fold,
            label_map,
            batch_size=config["training"]["batch_size"]
        )

        optimizer = get_optimizer(
            model,
            config["training"]["optimizer"],
            **config["training"]["optimizer_kwargs"],
        )
        loss_fn = get_loss_function(
            config["training"]["loss"],
            device=device,
            **config["training"]["loss_kwargs"],
        )
        # Initialize the trainer
        trainer = EmbeddingMILTrainer(
            model,
            wsi_dataloaders,
            optimizer,
            loss_fn,
            device=device,
            hdf5_file=hdf5_file,
        )

        # Train the model
        logger.info(f"Starting training for fold {fold+1}")
        trainer.train(num_epochs)
        logger.info(f"Completed training for fold {fold+1}")
        # Save the model
        model_save_path = model_dir / f"weights/mil_model_fold_{fold+1}.pth"
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model for fold {fold+1} saved to {model_save_path}")

    hdf5_file.close()


if __name__ == "__main__":
    main()

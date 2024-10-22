from pathlib import Path
import json
import logging

import torch
import h5py
import pandas as pd
import pytorch_lightning as pl
from lightning.pytorch import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
# import mlflow

from histolung.models.models import MILModel
from histolung.mil.lightning_module import MILLightningModule
from histolung.mil.utils import (get_wsi_dataloaders, get_loss_function,
                                 get_optimizer)
from histolung.utils import yaml_load

# Set project directories and configuration paths
project_dir = Path(__file__).resolve().parents[2]
config_path = "/home/valentin/workspaces/histolung/models/MIL/uni/config.yaml"
model_dir = Path("/home/valentin/workspaces/histolung/models/MIL/uni/")


def set_up_logging(config):
    """
    Set up logging to a file based on the model directory. This configures the root logger,
    so all loggers in the codebase inherit this setup.
    """
    exp_name = config["run"]["experiment_name"]
    log_file = project_dir / f"logs/training/training_from_embedding_{exp_name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Clear existing handlers to avoid duplication
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Log to a file
        ])

    # Log the initial setup
    logging.info(f"Logging set up for experiment: {exp_name}")


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # sets seeds for numpy, torch and python.random.
    seed_everything(42, workers=True)

    # Configuration
    config = yaml_load(config_path)
    set_up_logging(config)
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
            batch_size=config["training"]["batch_size"])

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

        mil_module = MILLightningModule(
            model=model,
            loss_fn=loss_fn,
            optimizer_config=config["training"]["optimizer_kwargs"],
            num_classes=config["model"]["num_classes"],
            hdf5_file=hdf5_file,
        )
        # Define Callbacks
        checkpoint_callback = ModelCheckpoint(dirpath=project_dir / "models/MIL/first/weights",
                                              monitor="val_loss",
                                              save_top_k=1,
                                              mode="min",)
        early_stopping_callback = EarlyStopping(monitor="val_loss",
                                                patience=5,
                                                mode="min",)

        # mlflow.set_experiment(config["run"]["experiment_name"])
        # with mlflow.start_run():
        # mlflow.log_params(config)

        # Initialize PyTorch Lightning Trainer
        trainer = Trainer(
            strategy=DDPStrategy(find_unused_parameters=True),
            max_epochs=num_epochs,
            devices=[0],
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=False  # No built-in logger, use MLflow
        )

        # Train the model
        trainer.fit(mil_module, wsi_dataloaders["train"],
                    wsi_dataloaders["val"])

        # Log the best model to MLflow
        # mlflow.pytorch.log_model(model, "model")

    hdf5_file.close()


if __name__ == "__main__":
    main()

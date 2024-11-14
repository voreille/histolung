from pathlib import Path
import logging

import torch

from histolung.models.models import MILModel
from histolung.mil.mil_trainer import TileMILTrainer
from histolung.mil.utils import (get_preprocessing, get_wsi_dataloaders,
                                 get_loss_function, get_optimizer,
                                 load_metadata, split_by_fold)
from histolung.utils import yaml_load

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set project directories and configuration paths
project_dir = Path(__file__).resolve().parents[2]


def main(model_dir=None):
    # Configuration
    num_epochs = 1
    label_map = {"luad": 0, "lusc": 1}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data augmentation
    # I think it must be

    config = yaml_load(model_dir / "config.yaml")
    preprocess = get_preprocessing(config["data"])
    # Load metadata and split into k-folds
    logger.info("Loading metadata...")
    wsi_metadata, fold_df = load_metadata(project_dir)
    tile_paths_by_wsi = {d["wsi_id"]: d["patch_files"] for d in wsi_metadata}
    wsi_metadata_by_folds, n_folds = split_by_fold(wsi_metadata, fold_df)

    # Loop over each fold
    for fold in range(n_folds):
        logger.info(f"Processing fold {fold+1}/{n_folds}")

        # Load the model from configuration
        logger.info("Loading model configuration...")
        model = MILModel.from_config(config)

        # Get the DataLoaders for the current fold
        logger.info(f"Creating dataloaders for fold {fold+1}")
        wsi_dataloaders = get_wsi_dataloaders(
            wsi_metadata_by_folds,
            fold,
            label_map,
        )

        # Define the optimizer and loss function
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
        trainer = TileMILTrainer(
            model,
            wsi_dataloaders,
            optimizer,
            loss_fn,
            device=device,
            training_cfg=config["training"],
            tile_paths_by_wsi=tile_paths_by_wsi,
            tile_preprocess=preprocess,
        )

        # Train the model
        logger.info(f"Starting training for fold {fold+1}")
        trainer.train(num_epochs)
        logger.info(f"Completed training for fold {fold+1}")
        # Save the model
        model_save_path = model_dir / f"mil_model_fold_{fold+1}.pth"
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model for fold {fold+1} saved to {model_save_path}")


if __name__ == "__main__":
    model_dir = project_dir / "models/MIL/uni/"
    main(model_dir)

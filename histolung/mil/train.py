from pathlib import Path
import json
import logging

import torch
from torch.nn import BCEWithLogitsLoss
import torchvision.transforms as T
from torch.optim import Adam, SGD, AdamW, RMSprop
from torch.utils.data import DataLoader
import pandas as pd

from histolung.mil.data_loader import WSIDataset
from histolung.models.models_refactor import MILModel
from histolung.mil.mil_trainer import MILTrainer
from histolung.mil.loss import FocalBCEWithLogitsLoss
from histolung.utils import yaml_load

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set project directories and configuration paths
project_dir = Path(__file__).resolve().parents[2]


def load_metadata():
    """
    Load WSI metadata and fold information for cross-validation.
    """
    fold_df = pd.read_csv(project_dir / "data/interim/tcga_folds.csv")
    with open(project_dir / "data/interim/tcga_wsi_data.json") as f:
        wsi_metadata = json.load(f)
    return wsi_metadata, fold_df


def split_by_fold(wsi_metadata, fold_df):
    """
    Split WSI metadata by fold for k-fold cross-validation.
    """
    n_folds = fold_df["fold"].max() + 1
    output = [[] for _ in range(n_folds)]
    fold_mapping = dict(zip(fold_df['wsi_id'], fold_df['fold']))

    for wsi_info in wsi_metadata:
        fold = fold_mapping.get(wsi_info['wsi_id'])
        if fold is not None:
            output[fold].append(wsi_info)

    return output, n_folds


def get_wsi_dataloaders(wsi_metadata_by_folds, fold, label_map, batch_size=2):
    """
    Create training and validation DataLoaders based on the current fold.
    """
    wsi_meta_train = [
        wsi for i, wsi_fold in enumerate(wsi_metadata_by_folds) if i != fold
        for wsi in wsi_fold
    ]
    wsi_meta_val = wsi_metadata_by_folds[fold]

    train_dataset = WSIDataset(
        wsi_meta_train,
        label_map=label_map,
    )
    val_dataset = WSIDataset(
        wsi_meta_val,
        label_map=label_map,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "val": val_loader}


def get_preprocessing(data_cfg):
    image_size = data_cfg["image_size"]
    mean = data_cfg["mean"]
    std = data_cfg["std"]
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_optimizer(net, optimizer_name, **kwargs):
    optimizer_dict = {
        "Adam": Adam,
        "AdamW": AdamW,
        "SGD": SGD,
        "RMSprop": RMSprop
    }

    logging.info(f"== Optimizer: {optimizer_name} ==")

    optimizer_class = optimizer_dict.get(optimizer_name)

    if optimizer_class is None:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported")

    return optimizer_class(net.parameters(), **kwargs)


def get_loss_function(loss_name, device="gpu", **kwargs):
    loss_dict = {
        "BCEWithLogitsLoss": BCEWithLogitsLoss,
        "FocalBinaryCrossEntropy": FocalBCEWithLogitsLoss,
    }

    loss_class = loss_dict.get(loss_name)

    if loss_class is None:
        raise ValueError(f"Loss function '{loss_name}' is not supported.\n"
                         f"The available losses are: {list(loss_dict.keys())}")

    logging.info(f"Using loss function: {loss_name} with arguments: {kwargs}")

    # Check if 'weight' is in kwargs and convert it to a tensor
    if 'weight' in kwargs and not isinstance(kwargs['weight'], torch.Tensor):
        kwargs['weight'] = torch.tensor(kwargs['weight'],
                                        dtype=torch.float,
                                        device=device)

    return loss_class(**kwargs)


def main(model_dir=None):
    # Configuration
    num_epochs = 1
    label_map = {"luad": 0, "lusc": 1}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data augmentation
    # I think it must be

    config = yaml_load(model_dir / "config.yml")
    preprocess = get_preprocessing(config["data"])
    # Load metadata and split into k-folds
    logger.info("Loading metadata...")
    wsi_metadata, fold_df = load_metadata()
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
        trainer = MILTrainer(
            model,
            wsi_dataloaders,
            optimizer,
            loss_fn,
            device=device,
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
    model_dir = project_dir / "models/MIL/first/"
    main(model_dir)

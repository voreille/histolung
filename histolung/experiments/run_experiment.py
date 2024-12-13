from pathlib import Path
import json

import h5py
import mlflow
import click
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn.functional as F

from histolung.models.pl_models import AggregatorPL
from histolung.mil.embedding_manager import EmbeddingManager
from histolung.mil.utils import get_preloadedembedding_dataloaders
from histolung.utils import yaml_load

project_dir = Path(__file__).resolve().parents[2]


# Utility functions
def load_and_validate_config(experiment_id):
    config_path = project_dir / f'histolung/experiments/config/{experiment_id}.yaml'
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}")
    return yaml_load(config_path)


def setup_logger(config):
    mlflow.set_experiment(config["run"]["experiment_name"])
    return MLFlowLogger(experiment_name=config["run"]['experiment_name'])


def load_data(hdf5_path, label_map=None, debug_max_samples=None):

    num_classes = len(label_map.keys())

    with open(project_dir / "data/interim/tcga_wsi_data.json") as f:
        wsi_metadata = json.load(f)

    with h5py.File(hdf5_path, "r") as file:
        wsi_ids_embedded = list(file["embeddings"].keys())

        outputs = [(k["wsi_id"], k["label"]) for k in wsi_metadata
                   if k["wsi_id"] in wsi_ids_embedded]
        wsi_ids, labels = zip(*outputs)

        if debug_max_samples:
            wsi_ids = wsi_ids[:debug_max_samples]
            labels = labels[:debug_max_samples]

        embeddings = {}
        labels_dict = {}
        for idx, wsi_id in enumerate(wsi_ids):
            embeddings[wsi_id] = torch.tensor(file['embeddings'][wsi_id][:])
            label = label_map[labels[idx]]
            labels_one_hot = F.one_hot(
                torch.tensor(label),
                num_classes=num_classes,
            )

            labels_dict[wsi_id] = labels_one_hot

    return wsi_ids, embeddings, labels, labels_dict


def load_data_as_list(hdf5_path, label_map, debug_max_samples=None):
    """
    Preload embeddings and labels into lists for efficient access.

    Args:
        hdf5_path (str): Path to the HDF5 file containing embeddings.
        label_map (dict): Mapping from label names to integer indices.
        debug_max_samples (int, optional): Maximum number of samples to load for debugging.

    Returns:
        tuple: wsi_ids (list), embeddings (list), labels_one_hot_list (list)
    """
    num_classes = len(label_map.keys())

    # Load WSI metadata
    with open(project_dir / "data/interim/tcga_wsi_data.json") as f:
        wsi_metadata = json.load(f)

    wsi_ids = []
    labels = []
    embeddings = []
    labels_one_hot_list = []

    # Open the HDF5 file and preload embeddings and labels
    with h5py.File(hdf5_path, "r") as hdf5_file:
        wsi_ids_embedded = set(hdf5_file["embeddings"].keys())

        for metadata in wsi_metadata:
            wsi_id = metadata["wsi_id"]
            if wsi_id in wsi_ids_embedded:
                label = metadata["label"]

                # Add to the list
                wsi_ids.append(wsi_id)
                labels.append(label)
                embeddings.append(
                    torch.tensor(hdf5_file["embeddings"][wsi_id][:]))
                labels_one_hot_list.append(
                    F.one_hot(torch.tensor(label_map[label]),
                              num_classes=num_classes))

                # Stop if debug_max_samples is reached
                if debug_max_samples and len(wsi_ids) >= debug_max_samples:
                    break

    return wsi_ids, embeddings, labels, labels_one_hot_list


def generate_embeddings_task(config, force=False, **kwargs):
    click.echo("Generating embeddings...")
    embedding_manager = EmbeddingManager(config, **kwargs)
    if force and embedding_manager.embeddings_exist():
        embedding_manager.delete_file()
    if not embedding_manager.embeddings_exist():
        embedding_manager.generate_embeddings(
            max_wsi_debug=kwargs.get('max_wsi_debug', -1))
    else:
        click.echo("Embeddings already exist. Skipping.")


def train_model(config,
                gpu_id,
                mlflow_logger=None,
                n_folds=5,
                debug_max_samples=None):
    click.echo("Training model...")
    embedding_manager = EmbeddingManager(config)
    hdf5_path = embedding_manager.get_embedding_path()
    wsi_ids, embeddings, labels, labels_one_hot = load_data_as_list(
        hdf5_path,
        label_map=config["data"]["label_map"],
        debug_max_samples=debug_max_samples,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(skf.split(wsi_ids,
                                                              labels)):
        click.echo(f"Processing fold {fold + 1}/{n_folds}")
        model = AggregatorPL.from_config(config)

        wsi_dataloader_train = get_preloadedembedding_dataloaders(
            wsi_ids,
            embeddings,
            labels_one_hot,
            train_index,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=0,
            prefetch_factor=None,
            pin_memory=True,
            resample=1000,
        )
        wsi_dataloader_val = get_preloadedembedding_dataloaders(
            wsi_ids,
            embeddings,
            labels_one_hot,
            val_index,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=0,
            prefetch_factor=None,
            pin_memory=True,
        )

        trainer = Trainer(
            max_epochs=config["training"]["epochs"],
            logger=mlflow_logger,
            deterministic=True,
            log_every_n_steps=10,
            accelerator="gpu",
            devices=[gpu_id],
        )
        trainer.fit(model, wsi_dataloader_train, wsi_dataloader_val)

        checkpoint_dir = Path(config["aggregator"]["checkpoints_dir"])
        checkpoint_path = checkpoint_dir / f"weights/mil_model_fold_{fold + 1}.ckpt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(checkpoint_path)
        click.echo(f"Model for fold {fold + 1} saved to {checkpoint_path}")


def evaluate_model(config, experiment_id):
    click.echo("Evaluating model...")
    checkpoint_path = Path(config['feature_aggregator']
                           ['checkpoints_dir']) / f"{experiment_id}.ckpt"
    if not checkpoint_path.exists():
        click.echo("Checkpoint not found. Train the model first.")
        return
    # Load and evaluate model here
    click.echo("Evaluation not yet implemented.")


# CLI commands
@click.group()
def cli():
    """CLI for managing experiments."""
    pass


@cli.command()
@click.option('--id', 'experiment_id', required=True, help='Experiment ID.')
@click.option('--force', is_flag=True, help='Force embedding regeneration.')
@click.option('--gpu-id', default=0, help='GPU ID for embedding generation.')
@click.option('--batch-size',
              default=32,
              help='Batch size for embedding generation.')
@click.option('--num-workers',
              default=8,
              help='Number of workers for dataloaders.')
def generate_embeddings(experiment_id, force, gpu_id, batch_size, num_workers):
    """Generate embeddings only."""
    config = load_and_validate_config(experiment_id)
    generate_embeddings_task(config,
                             force,
                             gpu_id=gpu_id,
                             batch_size=batch_size,
                             num_workers=num_workers)


@cli.command()
@click.option('--id', 'experiment_id', required=True, help='Experiment ID.')
@click.option('--gpu-id', default=0, help='GPU ID for embedding generation.')
@click.option('--n-folds',
              default=5,
              help='The number of fold for the training')
@click.option('--debug-max-samples',
              default=None,
              type=int,
              help='Maximum number of embedding loaded, for debugging purpose')
def train(experiment_id, gpu_id, n_folds, debug_max_samples):
    """Train model only."""
    config = load_and_validate_config(experiment_id)
    mlflow_logger = setup_logger(config)
    train_model(config,
                gpu_id,
                mlflow_logger=mlflow_logger,
                n_folds=n_folds,
                debug_max_samples=debug_max_samples)


@cli.command()
@click.option('--id', 'experiment_id', required=True, help='Experiment ID.')
def evaluate(experiment_id):
    """Evaluate model only."""
    config = load_and_validate_config(experiment_id)
    evaluate_model(config, experiment_id)


if __name__ == '__main__':
    cli()

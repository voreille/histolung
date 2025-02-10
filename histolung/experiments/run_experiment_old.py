import os
from pathlib import Path
import json

import mlflow
import click
import torch
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer
import pandas as pd

from histolung.models.pl_models import AggregatorPL
from histolung.mil.embedding_manager import EmbeddingManager
from histolung.mil.utils import get_embedding_dataloaders
from histolung.utils import yaml_load

project_dir = Path(__file__).resolve().parents[2]


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


def load_metadata():
    """Load WSI metadata and fold information for cross-validation."""
    fold_df = pd.read_csv(project_dir / "data/interim/tcga_folds.csv")
    with open(project_dir / "data/interim/tcga_wsi_data.json") as f:
        wsi_metadata = json.load(f)
    return wsi_metadata, fold_df


# Helper function to generate embeddings if they do not already exist
def generate_embeddings_if_needed(
    config,
    mlflow_logger=None,
    max_wsi_debug=-1,
    force=False,
    batch_size=32,
    num_workers=8,
    gpu_id=None,
):
    click.echo("Step 1: Generating Embeddings (if not already computed)...")
    embedding_manager = EmbeddingManager(config,
                                         gpu_id=gpu_id,
                                         batch_size=batch_size,
                                         num_workers=num_workers)

    if force and embedding_manager.embeddings_exist():
        embedding_manager.delete_file()

    if not embedding_manager.embeddings_exist():
        embedding_manager.generate_embeddings(max_wsi_debug=max_wsi_debug)
        if mlflow_logger:
            mlflow_logger.experiment.log_artifact(
                mlflow_logger.run_id, embedding_manager.get_embedding_path())
    else:
        click.echo("Embeddings already exist. Skipping embedding generation.")


def get_dataloaders(wsi_metadata_by_folds, fold, hdf5_path, batch_size=1):
    wsi_meta_train = [
        wsi for i, wsi_fold in enumerate(wsi_metadata_by_folds) if i != fold
        for wsi in wsi_fold
    ]
    wsi_meta_val = wsi_metadata_by_folds[fold]

    wsi_ids_train = [k for k in wsi_meta_train["wsi_id"]]
    labels_train = [k for k in wsi_meta_train["label"]]

    wsi_ids_val = [k for k in wsi_meta_val["wsi_id"]]
    labels_val = [k for k in wsi_meta_val["label"]]

    wsi_dataloader_train = get_embedding_dataloaders(
        wsi_ids_train,
        labels_train,
        hdf5_path,
        batch_size=batch_size,
        preloading=True,
        shuffle=True,
    )
    wsi_dataloader_val = get_embedding_dataloaders(
        wsi_ids_val,
        labels_val,
        hdf5_path,
        batch_size=batch_size,
        preloading=True,
        shuffle=False,
    )
    return {
        "train": wsi_dataloader_train,
        "validation": wsi_dataloader_val,
    }


def train_total(config):
    # Load configuration file

    # Set up MLFlow logging
    mlflow.set_experiment(config["run"]["experiment_name"])
    mlflow_logger = MLFlowLogger(experiment_name=config['experiment_name'])

    # Set up logging

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load metadata and split into k-folds
    wsi_metadata, fold_df = load_metadata()
    wsi_metadata_by_folds, n_folds = split_by_fold(wsi_metadata, fold_df)

    # Initialize EmbeddingManager
    embedding_manager = EmbeddingManager(config)

    # Open HDF5 file for reading embeddings
    hdf5_path = embedding_manager.get_embedding_path()

    # Loop over each fold
    for fold in range(n_folds):
        click.echo(f"Processing fold {fold + 1}/{n_folds}")

        # Load the model from configuration
        model = AggregatorPL.from_config(config)
        dataloaders = get_dataloaders(
            wsi_metadata_by_folds,
            fold,
            hdf5_path,
            batch_size=config["training"]["batch_size"],
        )
        trainer = Trainer(
            max_epochs=config["training"]["epochs"],
            logger=mlflow_logger,
            gpus=device,
            deterministic=True,
            log_every_n_steps=10,
        )
        # Train the model
        click.echo(f"Starting training for fold {fold + 1}")
        trainer.fit(model, dataloaders["train"], dataloaders["val"])
        click.echo(f"Completed training for fold {fold + 1}")

        # Save the model
        experiment_name = config["run"]["experiment_name"]
        model_save_path = (project_dir /
                           config["feature_aggregator"]["checkpoints_dir"] /
                           f"{experiment_name}" /
                           f"weights/mil_model_fold_{fold + 1}.ckpt")
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(model_save_path)
        click.echo(f"Model for fold {fold + 1} saved to {model_save_path}")


# Helper function to train the model if checkpoint does not already exist
def train_model_if_needed(config, experiment_id, mlflow_logger=None):
    click.echo("Step 1: Training the Model (if not already trained)...")
    checkpoint_path = os.path.join(config['checkpoint_dir'],
                                   f"{experiment_id}.ckpt")
    if not os.path.exists(checkpoint_path):
        click.echo(f"Training model for experiment {experiment_id}...")

        train_total(config)

        if mlflow_logger:
            mlflow_logger.experiment.log_artifact(mlflow_logger.run_id,
                                                  checkpoint_path)
    else:
        click.echo("Model checkpoint already exists. Skipping training.")


# Helper function to evaluate the model if a checkpoint exists
def evaluate_model(config, experiment_id, mlflow_logger=None):
    click.echo("Step 2: Evaluating the Model...")
    checkpoint_path = os.path.join(config['checkpoint_dir'],
                                   f"{experiment_id}.ckpt")
    if not os.path.exists(checkpoint_path):
        click.echo("No checkpoint found. Please train the model first.")
        return

    # Load the model and evaluate
    click.echo(f"Evaluating model for experiment {experiment_id}...")
    click.echo(f"But not yet implemented...")


@click.group()
def cli():
    """CLI for managing experiments involving embedding generation, training, and evaluation."""
    pass


@cli.command()
@click.option('--id',
              'experiment_id',
              required=True,
              help='ID of the experiment to run.')
def run(experiment_id):
    """Run an entire experiment: generate embeddings, train the model, and evaluate it."""

    # Load configuration file
    config_path = f'./experiments/config/{experiment_id}.yaml'
    config = yaml_load(config_path)

    # Set up MLFlow logging
    mlflow.set_experiment(config['experiment_name'])
    mlflow_logger = MLFlowLogger(experiment_name=config['experiment_name'])

    with mlflow.start_run(run_name=experiment_id) as run:
        mlflow.log_params(config)

        # Run all three steps in sequence: embeddings, training, evaluation
        train_model_if_needed(config, experiment_id, mlflow_logger)
        evaluate_model(config, experiment_id, mlflow_logger)


@cli.command()
@click.option('--id',
              'experiment_id',
              required=True,
              help='ID of the experiment.')
@click.option('--gpu-id', required=True, help='ID of the GPU.')
@click.option('--batch-size',
              required=False,
              default=32,
              help='ID of the GPU.')
@click.option('--num-workers',
              required=False,
              default=8,
              help='ID of the GPU.')
@click.option("--max-wsi-debug", default=-1)
@click.option("--force", is_flag=True)
def generate_embeddings(
    experiment_id,
    gpu_id,
    batch_size,
    num_workers,
    max_wsi_debug,
    force,
):
    """Generate embeddings for an experiment configuration if they do not exist."""
    config_path = project_dir / f"histolung/experiments/config/{experiment_id}.yaml"
    config = yaml_load(config_path)
    generate_embeddings_if_needed(
        config,
        max_wsi_debug=max_wsi_debug,
        force=force,
        gpu_id=gpu_id,
        batch_size=batch_size,
        num_workers=num_workers,
    )


@cli.command()
@click.option('--id',
              'experiment_id',
              required=True,
              help='ID of the experiment to train.')
def train(experiment_id):
    """Train a model using the specified experiment configuration."""
    config_path = project_dir / f'histolung/experiments/config/{experiment_id}.yaml'
    config = yaml_load(config_path)

    # Check if embeddings exist before training
    embedding_manager = EmbeddingManager(config)
    if not embedding_manager.embeddings_exist():
        click.echo(
            "Embeddings do not exist. Please generate embeddings first.")
        return

    # Start MLFlow logging
    mlflow.set_experiment(config['experiment_name'])
    mlflow_logger = MLFlowLogger(experiment_name=config['experiment_name'])

    with mlflow.start_run(run_name=experiment_id) as run:
        mlflow.log_params(config)
        train_model_if_needed(config, experiment_id, mlflow_logger)


@cli.command()
@click.option('--id',
              'experiment_id',
              required=True,
              help='ID of the experiment to evaluate.')
def evaluate(experiment_id):
    """Evaluate a model using the specified experiment configuration."""
    config_path = f'./experiments/config/{experiment_id}.yaml'
    config = yaml_load(config_path)
    evaluate_model(config, experiment_id)


if __name__ == '__main__':
    cli()

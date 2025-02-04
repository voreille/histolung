import click
import logging
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import yaml

from histolung.utils.yaml import load_yaml_with_env
from histolung.data.histoqc import run_histoqc
from histolung.data.rename import rename_masks_with_copy, write_wsi_paths_to_csv
from histolung.data.tiling import tile_dataset

logger = logging.getLogger(__name__)

# Base directory and configuration loading
project_dir = Path(__file__).parents[2].resolve()


def load_stuff(config=None):
    if config is None:
        config = load_yaml_with_env(project_dir /
                                    "histolung/config/datasets_config.yaml")

    if Path(config) == Path or type(config) == str:
        config = load_yaml_with_env(config)

    masks_basedir = project_dir / config["histoqc_masks_basedir"]
    tiles_basedir = project_dir / config["tiles_basedir"]
    return config, masks_basedir, tiles_basedir


def configure_task_logger(task_name, dataset=None, debug_id=None):
    """Set up logging with dynamic file path and format."""
    logs_dir = project_dir / "logs/data"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Include date, time, and task-specific information in the log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{timestamp}__{task_name}"
    if dataset:
        log_filename += f"__{dataset}"
    if debug_id:
        log_filename += "__debug"
    log_filename += ".log"

    log_path = logs_dir / log_filename

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Stream handler (for console output)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(asctime)s - %(message)s")
    stream_handler.setFormatter(stream_formatter)

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    root_logger.info(
        f"Logging configured for task '{task_name}' (dataset: {dataset}).")


@click.group()
def cli():
    """Preprocess datasets with various tasks"""
    pass


def parse_histoqc_config_mapping(filepath, input_dir):
    """
    Parse the HistoQC config mapping YAML file and resolve wsi_ids to file paths in the input directory,
    filtering only valid WSI files (e.g., .svs, .tiff).

    Args:
        filepath (str): Path to the YAML file containing mappings.
        input_dir (Path): Path to the directory containing the input WSIs.

    Returns:
        list[dict]: List of dictionaries, each containing 'config' and 'wsi_paths'.
    """
    valid_extensions = {
        ".svs", ".tif", ".tiff", ".ndpi", ".vms", ".vmu", ".scn", ".mrxs",
        ".bif", ".svslide"
    }

    with open(filepath, 'r') as f:
        config_mapping = yaml.safe_load(f)

    mappings = config_mapping["mappings"]
    resolved_mappings = []

    for mapping in mappings:
        config_path = Path(mapping["config"])
        wsi_ids = mapping["wsi_ids"]

        # Search for matching WSI files in the input directory
        wsi_paths = []
        for wsi_id in wsi_ids:
            matching_files = [
                file for file in input_dir.rglob(f"*{wsi_id}*")
                if file.suffix.lower() in valid_extensions
            ]
            if not matching_files:
                logger.warning(
                    f"No valid WSI files found for WSI ID: {wsi_id}")
            else:
                wsi_paths.extend(matching_files)

        resolved_mappings.append({
            "config": config_path,
            "wsi_paths": wsi_paths,
        })

    return resolved_mappings


@cli.command()
@click.option("--dataset",
              required=True,
              help="Dataset name (e.g., 'tcga_luad')")
@click.option("--num_workers",
              default=1,
              show_default=True,
              help="Number of workers for parallel processing")
@click.option("-c",
              "--config",
              type=click.Path(exists=True),
              help="Path to the configuration file",
              default=None)
def run_histoqc_task(dataset, num_workers, config):
    """Run HistoQC on the specified dataset."""
    config, masks_basedir, _ = load_stuff(config)
    configure_task_logger("histoqc", dataset)
    dataset_config = config["datasets"].get(dataset)
    logger.info(f"Using dataset_config: {dataset_config}")

    if not dataset_config:
        raise click.BadParameter(
            f"Dataset '{dataset}' not found in configuration.")

    input_dir = Path(dataset_config["data_dir"])
    masks_dir = masks_basedir / dataset
    masks_dir.mkdir(parents=True, exist_ok=True)

    histoqc_config_mapping = dataset_config.get("histoqc_config_mapping")

    # Case 1: Default config for all files
    if histoqc_config_mapping is None:
        file_list = list(input_dir.glob(dataset_config["input_pattern"]))
        config_path = project_dir / dataset_config[
            "default_histoqc_config_path"]
        config_list = [config_path] * len(file_list)
    else:
        # Case 2 & 3: Resolve mappings from YAML
        histoqc_config_mapping = parse_histoqc_config_mapping(
            histoqc_config_mapping, input_dir)

        # Flatten file and config lists
        file_list = []
        config_list = []
        for mapping in histoqc_config_mapping:
            file_list.extend(mapping["wsi_paths"])
            config_list.extend([project_dir / mapping["config"]] *
                               len(mapping["wsi_paths"]))

    # Log the files and configs being processed
    logger.info(f"Files to process: {file_list}")
    logger.info(f"Configurations: {config_list}")

    # Run HistoQC
    logger.info(f"Running HistoQC for dataset: {dataset}")
    run_histoqc(
        file_list,
        config_list,
        input_dir,
        masks_dir,
        user=None,
        force=False,
        num_workers=num_workers,
    )
    logger.info("HistoQC completed.")


@cli.command()
@click.option("--dataset",
              required=True,
              help="Dataset name (e.g., 'tcga_luad')")
@click.option("--num_workers",
              default=1,
              show_default=True,
              help="Number of workers for parallel processing")
@click.option("-c",
              "--config",
              type=click.Path(exists=True),
              help="Path to the configuration file",
              default=None)
def run_superpixel_segmentation(dataset, num_workers, config):
    """Run HistoQC on the specified dataset."""
    config, masks_basedir, _ = load_stuff(config)
    configure_task_logger("histoqc", dataset)
    dataset_config = config["datasets"].get(dataset)
    logger.info(f"Using dataset_config: {dataset_config}")

    if not dataset_config:
        raise click.BadParameter(
            f"Dataset '{dataset}' not found in configuration.")

    input_dir = Path(dataset_config["data_dir"])
    masks_dir = masks_basedir / dataset
    masks_dir.mkdir(parents=True, exist_ok=True)

    histoqc_config_mapping = dataset_config.get("histoqc_config_mapping")

@cli.command()
@click.option("--dataset",
              required=True,
              help="Dataset name (e.g., 'tcga_luad')")
@click.option("-c",
              "--config",
              type=click.Path(exists=True),
              help="Path to the configuration file",
              default=None)
def rename_masks_task(dataset, config):
    """Rename mask folders according to dataset conventions"""
    configure_task_logger("rename_masks_task", dataset)
    config, masks_basedir, _ = load_stuff(config)
    dataset_config = config["datasets"].get(dataset)
    if not dataset_config:
        raise click.BadParameter(
            f"Dataset '{dataset}' not found in configuration.")

    masks_dir = masks_basedir / dataset
    logger.info(f"Renaming masks for dataset: {dataset}")
    rename_masks_with_copy(masks_dir, dataset)
    logger.info("Mask renaming completed.")

    logger.info(f"Writing raw WSI path to CSV")
    write_wsi_paths_to_csv(
        raw_data_dir=Path(dataset_config["data_dir"]).resolve(),
        masks_dir=masks_dir,
        output_csv=masks_dir / "raw_wsi_path.csv",
        dataset=dataset,
    )

    logger.info(f"Writing raw WSI path to CSV completed")


@cli.command()
@click.option("-c",
              "--config",
              type=click.Path(exists=True),
              help="Path to the configuration file",
              default=None)
def write_tiles_metadata(config):
    """Store metatdata about tiles path, label for the dataloader"""
    config, _, tiles_basedir = load_stuff(config)
    tiles_basedir = Path(config["tiles_basedir"])
    preprocessed_datasets = [
        f.name for f in tiles_basedir.iterdir() if f.is_dir()
    ]
    wsi_data = []
    for dataset in preprocessed_datasets:
        dataset_config = config["datasets"].get(dataset)
        tiled_data_dir = tiles_basedir / dataset
        label = dataset_config.get("label")
        for wsi_id_dir in tiled_data_dir.iterdir():
            if wsi_id_dir.is_dir():
                patch_dir = wsi_id_dir / "tiles"
                if patch_dir.exists():
                    patch_files = [str(p) for p in patch_dir.glob("*.png")]
                    if len(patch_files) == 0:
                        logger.warning(f"WSI in dataset {dataset} and with "
                                       f"ID {wsi_id_dir.name} has not tiles "
                                       f"thus it has been discarded.")
                        continue
                    wsi_data.append({
                        "wsi_id": wsi_id_dir.name,
                        "label": label,
                        "patch_dir": str(patch_dir),
                        "patch_files": patch_files,
                    })
    output_json = tiles_basedir / "tiles_metadata.json"
    with open(output_json, 'w') as f:
        json.dump(wsi_data, f, indent=4)

    logger.info(f"Writing tiles metadata to {output_json}")


@cli.command()
@click.option(
    "--dataset",
    required=True,
    help="Dataset name (e.g., 'tcga_luad')",
)
@click.option(
    "--tile_size",
    type=click.INT,
    default=None,
    help="Tile size for tiling the WSI images",
)
@click.option(
    "--threshold",
    type=click.FLOAT,
    default=None,
    help="Threshold for tile inclusion based on mask coverage",
)
@click.option(
    "--magnification",
    type=click.INT,
    default=None,
    help="magnification for tiling the WSI images",
)
@click.option(
    "--num_workers",
    type=click.INT,
    default=1,
    show_default=True,
    help="Number of workers for parallel processing",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    help="Path to the configuration file",
    default=None,
)
@click.option("--debug_id", default=None, help="WSI ID used for debugging.")
def tile_wsi_task(
    dataset,
    tile_size,
    threshold,
    magnification,
    num_workers,
    config,
    debug_id,
):
    """Tile WSIs for the specified dataset"""
    # if debug_id is not None:

    config, masks_basedir, tiles_basedir = load_stuff(config)

    if debug_id:
        configure_task_logger("tile_wsi_task", dataset + "__debug")
    else:
        configure_task_logger("tile_wsi_task", dataset)

    if tile_size is None:
        tile_size = config["parameters"]["tile_size"]

    if threshold is None:
        threshold = config["parameters"]["tile_coverage"]

    if magnification is None:
        magnification = config["parameters"]["magnification"]

    dataset_config = config["datasets"].get(dataset)
    if not dataset_config:
        raise click.BadParameter(
            f"Dataset '{dataset}' not found in configuration.")

    # data_dir = Path(dataset_config["data_dir"])
    masks_dir = masks_basedir / dataset
    if debug_id:
        _tiles_basedir = tiles_basedir.parent / f"debug_{tiles_basedir.name}"
    else:
        _tiles_basedir = tiles_basedir

    tiles_dir = _tiles_basedir / dataset
    tiles_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Tiling WSIs for dataset: {dataset}")
    tile_dataset(masks_dir,
                 tiles_dir,
                 tile_size=tile_size,
                 threshold=threshold,
                 num_workers_tiles=4,
                 num_workers_wsi=num_workers,
                 save_tile_overlay=True,
                 debug_id=debug_id,
                 magnification=magnification,
                 save_masks=config["parameters"].get("save_masks", False))
    logger.info("WSI tiling completed.")


@cli.command()
@click.option("--dataset",
              required=True,
              help="Dataset name to process (e.g., 'tcga_luad')")
def process_dataset(dataset):
    """Run full preprocessing pipeline for a dataset"""

    logger.info(f"Starting full preprocessing for dataset: {dataset}")
    configure_task_logger("process_dataset", dataset)

    # Run all tasks in sequence for the dataset
    ctx = click.get_current_context()
    ctx.invoke(run_histoqc_task, dataset=dataset)
    ctx.invoke(rename_masks_task, dataset=dataset)
    ctx.invoke(tile_wsi_task, dataset=dataset)
    ctx.invoke(write_tiles_metadata)

    logger.info(f"Completed preprocessing for dataset: {dataset}")


if __name__ == "__main__":
    cli()

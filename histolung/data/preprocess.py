import click
import logging
from pathlib import Path
import json
from datetime import datetime

from histolung.utils.yaml import load_yaml_with_env
from histolung.data.histoqc import run_histoqc
from histolung.data.rename import rename_masks_with_copy, write_wsi_paths_to_csv
from histolung.data.tiling import tile_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory and configuration loading
project_dir = Path(__file__).parents[2].resolve()
config_path = project_dir / "histolung/config/datasets_config.yaml"
config = load_yaml_with_env(config_path)

masks_basedir = project_dir / config["histoqc_masks_basedir"]
tiles_basedir = project_dir / config["tiles_basedir"]


def configure_task_logger(task_name, dataset=None):
    """Configure logging to a file specific to the task and dataset."""
    logs_dir = project_dir / "logs/data"
    logs_dir.mkdir(parents=True, exist_ok=True)

    current_date = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"{current_date}_{task_name}"
    if dataset:
        log_filename += f"__{dataset}"
    log_filename += ".log"

    log_path = logs_dir / log_filename
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Clear previous handlers and set the file handler
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())

    logger.info(
        f"Logging configured for task '{task_name}' with dataset '{dataset}'.")


@click.group()
def cli():
    """Preprocess datasets with various tasks"""
    pass


@cli.command()
@click.option("--dataset",
              required=True,
              help="Dataset name (e.g., 'tcga_luad')")
@click.option("--num_workers",
              default=1,
              show_default=True,
              help="Number of workers for parallel processing")
def run_histoqc_task(dataset, num_workers):
    """Run HistoQC on the specified dataset"""
    configure_task_logger("histoqc", dataset)
    dataset_config = config["datasets"].get(dataset)
    logger.info(f"Using data_config: {dataset_config}")

    if not dataset_config:
        raise click.BadParameter(
            f"Dataset '{dataset}' not found in configuration.")

    input_dir = Path(dataset_config["data_dir"])
    masks_dir = masks_basedir / dataset
    masks_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running HistoQC for dataset: {dataset}")
    run_histoqc(input_dir,
                masks_dir,
                input_pattern=dataset_config["input_pattern"],
                user=None,
                config_path=project_dir /
                dataset_config["histoqc_config_path"],
                force=False,
                num_workers=num_workers)
    logger.info("HistoQC completed.")


@cli.command()
@click.option("--dataset",
              required=True,
              help="Dataset name (e.g., 'tcga_luad')")
def rename_masks_task(dataset):
    """Rename mask folders according to dataset conventions"""
    configure_task_logger("rename_masks_task", dataset)
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
        masks_dir / "results.tsv",
        masks_dir / "raw_wsi_path.csv",
        dataset,
        dataset_config,
    )
    logger.info(f"Writing raw WSI path to CSV completed")


@cli.command()
def write_tiles_metadata():
    """Store metatdata about tiles path, label for the dataloader"""
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
@click.option("--dataset",
              required=True,
              help="Dataset name (e.g., 'tcga_luad')")
@click.option("--tile_size",
              default=224,
              help="Tile size for tiling the WSI images")
@click.option("--threshold",
              default=0.8,
              help="Threshold for tile inclusion based on mask coverage")
@click.option("--num_workers",
              default=1,
              show_default=True,
              help="Number of workers for parallel processing")
@click.option("--debug_id", default=None, help="WSI ID used for debugging.")
def tile_wsi_task(
    dataset,
    tile_size,
    threshold,
    num_workers,
    debug_id=None,
):
    """Tile WSIs for the specified dataset"""
    # if debug_id is not None:
    if debug_id:
        configure_task_logger("tile_wsi_task", dataset + "__debug")
    else:
        configure_task_logger("tile_wsi_task", dataset)

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
    tile_dataset(
        masks_dir,
        tiles_dir,
        tile_size=tile_size,
        threshold=threshold,
        num_workers_tiles=4,
        num_workers_wsi=num_workers,
        save_tile_overlay=True,
        debug_id=debug_id,
        magnification=dataset_config["magnification"],
    )
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

import click
import logging
from pathlib import Path

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
    dataset_config = config["datasets"].get(dataset)
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
@click.option('--save_tile_overlay',
              is_flag=True,
              help="save a PNG showing the selected tiles")
@click.option("--debug_id", default=None, help="WSI ID used for debugging.")
def tile_wsi_task(
    dataset,
    tile_size,
    threshold,
    num_workers,
    save_tile_overlay,
    debug_id=None,
):
    """Tile WSIs for the specified dataset"""
    dataset_config = config["datasets"].get(dataset)
    if not dataset_config:
        raise click.BadParameter(
            f"Dataset '{dataset}' not found in configuration.")

    data_dir = Path(dataset_config["data_dir"])
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
        num_workers=num_workers,
        save_tile_overlay=save_tile_overlay,
        debug_id=debug_id,
    )
    logger.info("WSI tiling completed.")


@cli.command()
@click.option("--dataset",
              required=True,
              help="Dataset name to process (e.g., 'tcga_luad')")
def process_dataset(dataset):
    """Run full preprocessing pipeline for a dataset"""
    logger.info(f"Starting full preprocessing for dataset: {dataset}")

    # Run all tasks in sequence for the dataset
    ctx = click.get_current_context()
    ctx.invoke(run_histoqc_task, dataset=dataset)
    ctx.invoke(rename_masks_task, dataset=dataset)
    ctx.invoke(tile_wsi_task, dataset=dataset)

    logger.info(f"Completed preprocessing for dataset: {dataset}")


if __name__ == "__main__":
    cli()

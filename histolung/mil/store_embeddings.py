import logging
import time
import click
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import h5py
import json
from tqdm import tqdm

from histolung.models.models import MILModel
from histolung.mil.data_loader import WSIDataset, TileDataset
from histolung.utils import yaml_load

project_dir = Path(__file__).parents[2]


def set_up_logging(config):
    """
    Set up logging to a file based on the model directory.
    """
    exp_name = config["run"]["experiment_name"]
    log_file = project_dir / f"logs/data/embedding_process_exp_{exp_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Log to a file
        ])
    logging.info(f"Logging set up for model: {config}")


def load_metadata():
    """
    Load WSI metadata and fold information for cross-validation.
    """
    with open(project_dir / "data/interim/tcga_wsi_data.json") as f:
        wsi_metadata = json.load(f)

    logging.info("Loaded WSI metadata")
    return wsi_metadata


def get_wsi_dataloader(wsi_metadata, label_map, batch_size=1):
    """
    Create training and validation DataLoaders based on the current fold.
    """
    dataset = WSIDataset(
        wsi_metadata,
        label_map=label_map,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    logging.info("Created WSI DataLoader with batch size %d", batch_size)
    return dataloader


def store_all_embeddings(
    config,
    gpu_id,
    num_workers=4,
    tile_batch_size=32,
    wsi_batch_size=1,
    debug=False,
    max_wsi_debug=4,
):
    """
    Stores embeddings for all WSIs into a single HDF5 file.

    Args:
        model_dir (Path): Path to the model directory.
        gpu_id (int): GPU ID to use.
        num_workers (int): Number of workers for DataLoader.
        tile_batch_size (int): Batch size for processing tiles.
        wsi_batch_size (int): Batch size for WSIs.
        debug (bool): Whether to run in debug mode (limits the number of WSIs).
        max_wsi_debug (int): Maximum number of WSIs to process in debug mode.
    """
    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    output_file = Path(config["data"]["embedding_file"])
    output_file.parent.mkdir(exist_ok=True)

    wsi_metadata = load_metadata()

    wsi_dataloader = get_wsi_dataloader(
        wsi_metadata,
        label_map=config["data"]["label_map"],
        batch_size=wsi_batch_size,
    )
    tile_paths_by_wsi = {d["wsi_id"]: d["patch_files"] for d in wsi_metadata}

    model = MILModel.from_config(config)
    model.to(device)
    model.eval()  # Set model to evaluation mode

    preprocess = model.get_preprocessing(config["data"])

    start_time = time.time()  # Start measuring time

    wsi_count = 0  # Track number of WSIs processed
    # Create or open the HDF5 file
    with h5py.File(output_file, 'w') as hdf5_file:
        embeddings_group = hdf5_file.create_group(
            'embeddings')  # Optional group
        for wsi_ids, _ in tqdm(wsi_dataloader, desc="Processing WSIs"):
            for wsi_id in wsi_ids:
                logging.info(f"Processing WSI {wsi_id}")
                tile_paths = tile_paths_by_wsi[wsi_id]
                tile_loader = DataLoader(
                    TileDataset(
                        tile_paths,
                        preprocess=preprocess,
                    ),
                    batch_size=tile_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )

                # Compute embeddings for the tiles of this WSI
                embeddings = model.embed_with_dataloader(tile_loader)

                # Store the embeddings in the HDF5 file under the WSI ID
                embeddings_group.create_dataset(wsi_id,
                                                data=embeddings.cpu().numpy())

                logging.info(f"Stored embeddings for WSI {wsi_id}")

                # Increment the WSI count
                wsi_count += 1

                # If in debug mode and max WSI limit reached, stop early
                if debug and wsi_count >= max_wsi_debug:
                    logging.info(
                        f"DEBUG mode: Stopped after processing {wsi_count} WSIs."
                    )
                    break

            # Stop the outer loop if DEBUG mode limit is reached
            if debug and wsi_count >= max_wsi_debug:
                break

    total_time = time.time() - start_time  # End of time measurement
    logging.info(f"Stored all embeddings in {output_file}")
    logging.info(f"Total time taken: {total_time:.2f} seconds")


@click.command()
@click.option(
    '--config',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the model directory.",
)
@click.option('--gpu-id', type=int, default=0, help="GPU ID to use.")
@click.option(
    '--num-workers',
    type=int,
    default=4,
    help="Number of workers for DataLoader.",
)
@click.option(
    '--tile-batch-size',
    type=int,
    default=32,
    help="Batch size for processing tiles.",
)
@click.option(
    '--wsi-batch-size',
    type=int,
    default=1,
    help="Batch size for WSIs.",
)
@click.option(
    '--debug',
    is_flag=True,
    help="Enable debug mode to limit processing to 4 WSIs.",
)
def main(config, gpu_id, num_workers, tile_batch_size, wsi_batch_size, debug):
    """
    CLI for storing embeddings for all WSIs in a given model directory.
    """
    config = yaml_load(config)
    set_up_logging(config)
    logging.info("Starting the embedding process...")

    store_all_embeddings(
        config,
        gpu_id=gpu_id,
        num_workers=num_workers,
        tile_batch_size=tile_batch_size,
        wsi_batch_size=wsi_batch_size,
        debug=debug,
    )

    logging.info("Embedding process complete.")


if __name__ == "__main__":
    main()

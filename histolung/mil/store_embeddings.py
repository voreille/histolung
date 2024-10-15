from pathlib import Path
import logging
import time  # For time measurement

import torch
from torch.utils.data import DataLoader
import h5py
import json
from tqdm import tqdm

from histolung.models.models import MILModel
from histolung.mil.data_loader import WSIDataset, TileDataset
from histolung.mil.utils import get_preprocessing
from histolung.utils import yaml_load

project_dir = Path(__file__).parents[2]
# Set up logging to a file
model_path = project_dir / "models/MIL/uni"
log_file = project_dir / "logs/data/embedding_process_uni.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Log to a file
    ])

DEBUG = False  # Set to True to limit processing to 4 WSIs for testing
MAX_WSI_DEBUG = 4  # Limit for WSIs when debugging


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
    model,
    device='cuda',
    num_workers=4,
    tile_batch_size=32,
    wsi_batch_size=1,
):
    """
    Stores embeddings for all WSIs into a single HDF5 file.

    Args:
        model (MILModel): The MIL model used to generate embeddings.
        device (str): Device to use ('cuda' or 'cpu').
        num_workers (int): Number of workers for DataLoader.
        tile_batch_size (int): Batch size for processing tiles.
    """
    config_path = model_path / "config.yaml"
    output_file = model_path / "embeddings/all_embeddings.h5"
    output_file.parent.mkdir(exist_ok=True)
    config = yaml_load(config_path)

    logging.info(f"Loaded configuration from {config_path}")

    wsi_metadata = load_metadata()

    wsi_dataloader = get_wsi_dataloader(
        wsi_metadata,
        label_map={
            "lusc": 0,
            "luad": 1
        },
        batch_size=wsi_batch_size,
    )
    tile_paths_by_wsi = {d["wsi_id"]: d["patch_files"] for d in wsi_metadata}

    model = MILModel.from_config(config)
    model.to(device)
    model.eval()  # Set model to evaluation mode

    preprocess = get_preprocessing(config["data"])

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
                if DEBUG and wsi_count >= MAX_WSI_DEBUG:
                    logging.info(
                        f"DEBUG mode: Stopped after processing {wsi_count} WSIs."
                    )
                    break

            # Stop the outer loop if DEBUG mode limit is reached
            if DEBUG and wsi_count >= MAX_WSI_DEBUG:
                break

    total_time = time.time() - start_time  # End of time measurement
    logging.info(f"Stored all embeddings in {output_file}")
    logging.info(f"Total time taken: {total_time:.2f} seconds")


if __name__ == "__main__":
    # Store embeddings for all WSIs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # Start the embedding process and time it
    store_all_embeddings(model_path, device=device)

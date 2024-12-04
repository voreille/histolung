from pathlib import Path
import json
import time
import logging

import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from histolung.models.feature_extractor import BaseFeatureExtractor
from histolung.mil.data_loader import WSIDataset, TileDataset
from histolung.utils import yaml_load

project_dir = Path(__file__).resolve().parents[2]


class EmbeddingManager:

    def __init__(self, config, batch_size=32, gpu_id=None, num_workers=1):
        self.config = config
        self.embedding_file = Path(config['data']['embedding_file'])
        self.feature_extractor = None

        tiles_metadata_path = project_dir / config['data']['tiles_metadata']
        with open(tiles_metadata_path) as f:
            self.wsi_metadata = json.load(f)

        self.batch_size = config.get('tile_batch_size', 32)
        self.num_workers = config.get('num_workers', 4)
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.num_workers = num_workers

    def embeddings_exist(self):
        """
        Check if the embeddings HDF5 file already exists.
        """
        return self.embedding_file.exists()

    def delete_file(self):
        self.embedding_file.unlink()

    def get_embedding_path(self):
        """
        Get the path where embeddings are stored.
        """
        if not self.embeddings_exist():
            raise RuntimeError("Embeddings does not exist, "
                               "please run the command to generate it")
        return str(self.embedding_file)

    def generate_embeddings(self, max_wsi_debug=-1):
        """
        Generate embeddings for all WSIs and store them in an HDF5 file.
        """
        device = torch.device(
            f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # Ensure the output directory exists
        self.embedding_file.parent.mkdir(parents=True, exist_ok=True)

        # Load WSI DataLoader
        # wsi_dataloader = self._get_wsi_dataloader()
        wsi_ids = [d["wsi_id"] for d in self.wsi_metadata]
        tile_paths_by_wsi = {
            d["wsi_id"]: d["patch_files"]
            for d in self.wsi_metadata
        }

        # Load the model
        self.feature_extractor = BaseFeatureExtractor.from_config(self.config)
        self.feature_extractor.to(device)
        self.feature_extractor.eval()

        preprocess = self.feature_extractor.get_preprocessing(
            self.config["data"])

        start_time = time.time()  # Start measuring time

        wsi_count = 0  # Track number of WSIs processed

        # Create or open the HDF5 file
        with h5py.File(self.embedding_file, 'w') as hdf5_file:
            # Create groups for embeddings and metadata
            embeddings_group = hdf5_file.create_group('embeddings')
            metadata_group = hdf5_file.create_group('metadata')

            # Store configuration metadata
            metadata_group.attrs['config'] = json.dumps(self.config)
            metadata_group.attrs['feature_extractor_weights'] = str(
                Path(self.config['feature_extractor']['kwargs']
                     ['weights_filepath']))

            for wsi_id in tqdm(wsi_ids, desc="Processing WSIs"):
                logging.info(f"Processing WSI {wsi_id}")
                tile_paths = tile_paths_by_wsi[wsi_id]
                tile_loader = DataLoader(
                    TileDataset(tile_paths, preprocess=preprocess),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    prefetch_factor=2,
                    pin_memory=True,
                )

                # Compute embeddings for the tiles of this WSI
                embeddings = self.embed_with_dataloader(
                    tile_loader,
                    device=device,
                )

                # Store the embeddings in the HDF5 file under the WSI ID
                embeddings_group.create_dataset(wsi_id,
                                                data=embeddings.cpu().numpy())

                logging.info(f"Stored embeddings for WSI {wsi_id}")

                # Increment the WSI count
                wsi_count += 1

                # If in debug mode and max WSI limit reached, stop early
                if wsi_count >= max_wsi_debug and max_wsi_debug != -1:
                    logging.info(
                        f"DEBUG mode: Stopped after processing {wsi_count} WSIs."
                    )
                    break

        total_time = time.time() - start_time  # End of time measurement
        logging.info(f"Stored all embeddings in {self.embedding_file}")
        logging.info(f"Total time taken: {total_time:.2f} seconds")

    def embed_with_dataloader(self, dataloader, device=None):
        """
        Processes a WSI using a DataLoader, extracting embeddings in batches.
        
        Args:
            dataloader (DataLoader): DataLoader for batch-wise processing of patches.
        
        Returns:
            torch.Tensor: Extracted embeddings.
        """
        all_embeddings = []

        with torch.no_grad():
            for batch_patches in dataloader:
                batch_patches = batch_patches.to(device)
                embeddings = self.feature_extractor(batch_patches)
                all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def _get_wsi_dataloader(self):
        """
        Create DataLoader for WSI based on the current fold.
        """
        dataset = WSIDataset(self.wsi_metadata,
                             label_map=self.config["data"]["label_map"])
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )
        logging.info("Created WSI DataLoader with batch size %d",
                     self.config.get('batch_size', 1))
        return dataloader

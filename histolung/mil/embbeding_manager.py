from pathlib import Path

import h5py
import torch


class EmbeddingManager:

    def __init__(self,
                 features_hdf5_path,
                 model,
                 tile_paths_by_wsi,
                 tile_augmentation=None,
                 tile_preprocess=None,
                 device="cuda"):
        """
        Manages the saving, loading, and computation of embeddings for WSIs using HDF5 storage.
        
        Args:
            features_hdf5_path (str or Path): Path to the HDF5 file for storing/loading embeddings.
            model (MILModel): The MIL model to compute embeddings.
            tile_paths_by_wsi (dict): Mapping from WSI IDs to paths of their tile images.
            tile_augmentation (callable): Augmentation function for tile preprocessing.
            tile_preprocess (callable): Preprocessing function for tiles.
            device (str): Device on which to run the model and handle tensors ('cpu' or 'cuda').
        """
        self.features_hdf5_path = Path(features_hdf5_path)
        self.model = model.to(device)
        self.tile_paths_by_wsi = tile_paths_by_wsi
        self.tile_augmentation = tile_augmentation
        self.tile_preprocess = tile_preprocess
        self.device = device

        # Open the HDF5 file in append mode (create it if it doesn't exist)
        self.hdf5_file = None
        if not self.features_hdf5_path.exists():
            self.hdf5_file = h5py.File(self.features_hdf5_path, 'w')
        else:
            self.hdf5_file = h5py.File(self.features_hdf5_path, 'a')

    def close_hdf5(self):
        """Close the HDF5 file."""
        if self.hdf5_file:
            self.hdf5_file.close()

    def wsi_exists(self, wsi_id):
        """Check if embeddings for a WSI are already stored in the HDF5 file."""
        return wsi_id in self.hdf5_file

    def save_embeddings(self, embeddings, wsi_id):
        """
        Save the embeddings for a given WSI into the HDF5 file.
        
        Args:
            embeddings (torch.Tensor): The embeddings to save.
            wsi_id (str): The ID of the WSI.
        """
        if wsi_id in self.hdf5_file:
            del self.hdf5_file[wsi_id]  # Remove existing dataset if it exists
        self.hdf5_file.create_dataset(wsi_id, data=embeddings.cpu().numpy())

    def load_embeddings(self, wsi_id):
        """
        Load precomputed embeddings for the given WSI from the HDF5 file.
        
        Args:
            wsi_id (str): The ID of the WSI.
            
        Returns:
            torch.Tensor: The embeddings loaded from HDF5, or None if not found.
        """
        if self.wsi_exists(wsi_id):
            return torch.tensor(self.hdf5_file[wsi_id][:]).to(self.device)
        return None

    def compute_embeddings_for_wsi(self,
                                   wsi_id,
                                   batch_size=1024,
                                   num_workers=4):
        """
        Compute the embeddings for a given WSI by processing its tiles.
        
        Args:
            wsi_id (str): The ID of the WSI.
            batch_size (int): Batch size for processing the tiles.
            num_workers (int): Number of workers for the DataLoader.
            
        Returns:
            torch.Tensor: Computed embeddings for the WSI.
        """
        tile_dataset = TileDataset(
            self.tile_paths_by_wsi[wsi_id],
            augmentation=self.tile_augmentation,
            preprocess=self.tile_preprocess,
        )
        tile_loader = DataLoader(
            tile_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        embeddings = self.model.embed_with_dataloader(tile_loader)
        self.save_embeddings(embeddings, wsi_id)
        return embeddings

    def compute_embeddings_for_all(self, batch_size=1024, num_workers=4):
        """
        Compute embeddings for all WSIs if they are not already stored in the HDF5 file.
        
        Args:
            batch_size (int): Batch size for processing the tiles.
            num_workers (int): Number of workers for the DataLoader.
        """
        for wsi_id in self.tile_paths_by_wsi.keys():
            if not self.wsi_exists(wsi_id):
                print(f"Computing embeddings for WSI {wsi_id}...")
                self.compute_embeddings_for_wsi(wsi_id,
                                                batch_size=batch_size,
                                                num_workers=num_workers)
            else:
                print(
                    f"Embeddings for WSI {wsi_id} already exist, skipping...")

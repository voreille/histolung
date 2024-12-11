import torch
from torch.utils.data import Dataset
import pyspng
import h5py
import numpy as np


class WSIDataset(Dataset):

    def __init__(self, wsi_metadata: list, label_map: dict):
        """
        Initializes the WSI-level dataset to return WSI ID, patch file paths, and the class label.

        Args:
            wsi_metadata (list): List of dictionaries containing WSI metadata 
                                 (produced by `histolung.data.save_tiles_meta`), 
                                 where each dictionary contains 'wsi_id', 'patch_files', 
                                 and 'label'.
            label_map (dict):    Dictionary mapping string labels (e.g., 'luad', 'lusc') 
                                 to integer labels (e.g., 0, 1) for classification.
        """
        self.wsi_metadata = wsi_metadata  # Store the list of WSI metadata
        self.label_map = label_map

    def __len__(self):
        return len(self.wsi_metadata)  # Return the number of WSIs

    def __getitem__(self, idx: int):
        """
        Returns the WSI label (str), patch paths (list), and class label (int).
        
        Args:
            idx (int): Index of the WSI in the dataset.

        Returns:
            tuple: (wsi_id, patches_path, label)
                   wsi_id (str): The WSI ID.
                   patches_path (list): List of paths to tile images.
                   label (int): The class label (e.g., 0 for 'luad', 1 for 'lusc').
        """
        wsi_info = self.wsi_metadata[idx]
        wsi_id = wsi_info['wsi_id']
        label = self.label_map[wsi_info['label'].lower()]

        # Return WSI ID, list of patch paths, and label
        return wsi_id, label


class TileDataset(Dataset):

    def __init__(self, tile_paths, augmentation=None, preprocess=None):
        """
        Tile-level dataset that returns individual tile images from a list of paths.

        Args:
            tile_paths (list): List of paths to tile images for a WSI.
            augmentation (callable, optional): augmentation to apply to each tile image.
            transform (callable, optional): Transform to apply to each tile image.
        """
        self.tile_paths = tile_paths
        self.preprocess = preprocess
        self.augmentation = augmentation

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        with open(tile_path, 'rb') as f:
            image = pyspng.load(f.read())

        if self.augmentation:
            image = self.augmentation(image=image)['image']

        if self.preprocess:
            image = self.preprocess(image).type(torch.FloatTensor)

        return image


class HDF5EmbeddingDataset(Dataset):

    def __init__(
        self,
        hdf5_path=None,
        hdf5_file=None,
        label_map=None,
        wsi_ids=None,
    ):
        """
        Dataset for iterating through embeddings stored in an HDF5 file.
        
        Args:
            hdf5_path (str or Path): Path to the HDF5 file containing the embeddings.
            label_map (dict): Dictionary mapping `wsi_id` to class labels (int).
            wsi_ids (list): List of WSI IDs to iterate over.
        """
        self.hdf5_path = hdf5_path
        self.label_map = label_map  # Mapping between wsi_id and label
        self.wsi_ids = wsi_ids  # List of WSI IDs
        self.hdf5_file = hdf5_file  # Will hold the opened HDF5 file handle

    def __len__(self):
        """Return the number of WSIs in the dataset."""
        return len(self.wsi_ids)

    def open_hdf5(self):
        """Open the HDF5 file if not already opened."""
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')

    def close_hdf5(self):
        """Close the HDF5 file if it is open."""
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            self.hdf5_file = None

    def __getitem__(self, idx):
        """
        Fetch the embeddings and corresponding label for the WSI at index `idx`.
        
        Args:
            idx (int): Index of the WSI.
        
        Returns:
            tuple: (embeddings, label)
                   embeddings (torch.Tensor): Embeddings for the WSI.
                   label (int): The class label for the WSI.
        """
        self.open_hdf5()  # Ensure the HDF5 file is open

        wsi_id = self.wsi_ids[idx]
        embeddings = torch.tensor(self.hdf5_file['embeddings'][wsi_id][:])
        label = torch.tensor(self.label_map[wsi_id], dtype=torch.long)

        return embeddings, label

    def __del__(self):
        """Ensure the HDF5 file is closed when the dataset object is deleted."""
        self.close_hdf5()


class EmbeddingDataset(Dataset):
    """
    Dataset class for handling WSIs with an option for preloading embeddings.

    Args:
        hdf5_filepath (str): Path to the HDF5 file containing embeddings.
        wsi_ids (list of str): List of WSI IDs corresponding to the embeddings.
        labels (list of int): List of labels for each WSI.
        preloading (bool): If True, preload all embeddings into memory.
    """
    def __init__(self, hdf5_filepath, wsi_ids, labels, preloading=False):
        self.hdf5_filepath = hdf5_filepath
        self.wsi_ids = wsi_ids
        self.labels = labels
        self.preloading = preloading
        self.embeddings = None

        if self.preloading:
            self.embeddings = {}
            self._preload_embeddings()

    def _preload_embeddings(self):
        """Preload all embeddings into memory."""
        print("Preloading embeddings into memory...")
        with h5py.File(self.hdf5_filepath, 'r') as hdf5_file:
            for wsi_id in self.wsi_ids:
                self.embeddings[wsi_id] = torch.tensor(hdf5_file['embeddings'][wsi_id][:])
        print("Preloading complete.")

    def __len__(self):
        return len(self.wsi_ids)

    def __getitem__(self, idx):
        wsi_id = self.wsi_ids[idx]
        label = self.labels[idx]

        if self.preloading:
            embeddings = self.embeddings[wsi_id]
        else:
            # Load embeddings on demand
            with h5py.File(self.hdf5_filepath, 'r') as hdf5_file:
                embeddings = torch.tensor(hdf5_file['embeddings'][wsi_id][:])

        return wsi_id, embeddings, label
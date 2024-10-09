import numpy as np
import torch
from torch.utils.data import Dataset
import pyspng


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
        label = self.label_map[wsi_info['label']]

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

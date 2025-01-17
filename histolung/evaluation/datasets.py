from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import pyspng


def get_dataset_manager(dataset_name, metadata_df, transform=None):
    """
    Factory function to return the appropriate dataset manager.
    Args:
        dataset_name: Name of the dataset ('LungHist700', 'LC25000', etc.).
        metadata_df: DataFrame containing dataset metadata.
        transform: Transformations to apply to the dataset.
    Returns:
        An instance of the appropriate dataset manager.
    """
    dataset_managers = {
        'LungHist700': LungHist700DatasetManager,
        'LC25000': LC25000DatasetManager,
        'WSSS4LUAD': WSSS4LUADDatasetManager,
    }
    if dataset_name not in dataset_managers:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset_managers[dataset_name](metadata_df, transform=transform)


class BaseDatasetManager:
    REQUIRED_COLUMNS = ['tile_path', 'label']

    def __init__(self, metadata_df, transform=None, n_splits=5):
        self.metadata_df = metadata_df
        self.transform = transform
        self.n_splits = n_splits
        self.validate_metadata()
        self.tile_paths = metadata_df['tile_path'].values.tolist()

    def validate_metadata(self):
        """Ensure metadata contains required columns."""
        missing_columns = [
            col for col in self.REQUIRED_COLUMNS
            if col not in self.metadata_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Metadata is missing required columns: {missing_columns}")

    def get_folds(self, stratify_column='label', group_column=None):
        """
        Split the dataset into k folds and return train/validation datasets for each fold.
        Args:
            stratify_column: Column name to stratify by (e.g., 'label').
            group_column: Column name for grouping (e.g., 'patient_id'). If None, no grouping is applied.
        Returns:
            List of (train_dataset, validation_dataset) tuples.
        """
        splitter = (StratifiedGroupKFold(
            n_splits=self.n_splits, shuffle=True, random_state=42)
                    if group_column else StratifiedKFold(
                        n_splits=self.n_splits, shuffle=True, random_state=42))

        folds = []
        stratify = self.metadata_df[
            stratify_column] if stratify_column else None
        groups = self.metadata_df[group_column] if group_column else None

        for train_idx, val_idx in splitter.split(self.metadata_df, stratify,
                                                 groups):
            train_dataset = self.metadata_df.iloc[train_idx][
                'tile_id'].values.tolist()
            test_dataset = self.metadata_df.iloc[test_idx][
                'tile_id'].values.tolist()
            folds.append((train_dataset, test_dataset))

        return folds

    def get_splitter(self, stratify_column='label', group_column=None):
        """
        Split the dataset into k folds and return train/validation datasets for each fold.
        Args:
            stratify_column: Column name to stratify by (e.g., 'label').
            group_column: Column name for grouping (e.g., 'patient_id'). If None, no grouping is applied.
        Returns:
            List of (train_dataset, validation_dataset) tuples.
        """
        splitter = (StratifiedGroupKFold(
            n_splits=self.n_splits, shuffle=True, random_state=42)
                    if group_column else StratifiedKFold(
                        n_splits=self.n_splits, shuffle=True, random_state=42))

        folds = []
        stratify = self.metadata_df[
            stratify_column] if stratify_column else None
        groups = self.metadata_df[group_column] if group_column else None

        return splitter.split(self.metadata_df, stratify, groups)


class LungHist700DatasetManager(BaseDatasetManager):

    def get_folds(self):
        return super().get_folds(group_column='patient_id')


class LC25000DatasetManager(BaseDatasetManager):

    def get_folds(self):
        return super().get_folds(group_column=None)  # Standard CV


class WSSS4LUADDatasetManager(BaseDatasetManager):
    REQUIRED_COLUMNS = ['tile_path', 'label', 'mask_path']

    def get_folds(self):
        return super().get_folds()

    def validate_metadata(self):
        """Add validation for mask_path column."""
        super().validate_metadata()
        if 'mask_path' not in self.metadata_df.columns:
            raise ValueError("Metadata is missing the 'mask_path' column.")


class TileDataset(Dataset):

    def __init__(self, tile_paths, transform=None):
        """
        Dataset for histopathology tiles.
        Args:
            tile_paths: List or array of paths to tile images.
            labels: List or array of labels corresponding to the tiles.
            transform: Transformations to apply to the images.
        """
        self.tile_paths = tile_paths
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        image = pyspng.load(self.tile_paths[idx])  # Efficient PNG loading
        if self.transform:
            image = self.transform(image)
        return image, self.tile_paths[idx].stem


class EmbeddingDataset(Dataset):

    def __init__(self, embeddings, labels, transform=None):
        self.tile_paths = tile_paths
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        image = pyspng.load(self.tile_paths[idx])  # Efficient PNG loading
        if self.transform:
            image = self.transform(image)
        return image, self.tile_paths[idx].stem

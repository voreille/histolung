from pathlib import Path
from sklearn.model_selection import KFold


def prepare_kfold_tcga_data(root_dir, label_map, k=5):
    """
    Prepares data for k-fold cross-validation, keeping WSIs together and splitting at the WSI level.
    
    Args:
        root_dir (Path or str): Path to the root directory containing the WSIs and tiles.
        label_map (dict): Mapping from class names to labels.
        k (int): Number of folds for cross-validation.
        
    Returns:
        list: A list of (train_wsi, val_wsi) tuples for each fold, where train_wsi and val_wsi 
              are lists of (wsi_id, tiles, label).
    """
    root_dir = Path(root_dir)

    # Step 1: Collect all WSIs and their corresponding tiles and labels
    wsi_list = []  # List of (wsi_id, tiles, label)

    for class_name in ['luad', 'lusc']:
        class_dir = root_dir / f'tcga_{class_name}'
        for wsi_id_dir in class_dir.iterdir():
            if wsi_id_dir.is_dir():
                wsi_tiles_dir = wsi_id_dir / f'{wsi_id_dir.name}_tiles'
                if wsi_tiles_dir.exists():
                    tile_paths = list(wsi_tiles_dir.glob(
                        '*.png'))  # Get all tiles for the WSI
                    label = label_map[class_name]  # WSI-level label
                    wsi_list.append((wsi_id_dir.name, tile_paths,
                                     label))  # Append (wsi_id, tiles, label)

    # Step 2: Perform KFold splitting at the WSI level
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []

    for train_indices, val_indices in kf.split(wsi_list):
        # Collect WSIs for training and validation for the current fold
        train_wsi = [wsi_list[i] for i in train_indices]
        val_wsi = [wsi_list[i] for i in val_indices]

        folds.append((train_wsi, val_wsi))

    return folds

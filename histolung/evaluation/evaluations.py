import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

from histolung.models.feature_extractor import BaseFeatureExtractor
from histolung.evaluation.datasets import TileDataset

project_dir = Path(__file__).resolve().parents[2]

CONFIG = {
    "superclass_mapping": {
        "nor": 0,
        "aca": 1,
        "scc": 2
    },
    "debug": False,
    "embedding_path": "data/processed/LungHist700/embeddings.npz",
}


def compute_embeddings(model, dataloader, device="cuda"):
    embeddings = []
    tile_ids = []
    for batch in tqdm(dataloader):
        images, batch_tile_ids = batch
        embeddings.append(model(images.to(device)).detach().cpu())
        tile_ids.append(batch_tile_ids)
    embeddings = torch.cat(embeddings, dim=0)
    tile_ids = np.concatenate(tile_ids, axis=0)
    return embeddings, tile_ids


def compute_and_save_embeddings(model,
                                dataloader,
                                output_path=None,
                                device="cuda"):
    """
    Compute embeddings for a dataset and optionally save them to disk.
    Args:
        model: Torch model for feature extraction.
        dataloader: DataLoader for the dataset.
        output_path: Optional path to save embeddings and tile IDs.
        device: Device to run the model on ('cuda' or 'cpu').
    Returns:
        embeddings: NumPy array of embeddings.
        tile_ids: List of corresponding tile IDs.
    """
    model.eval()
    embeddings = []
    tile_ids = []

    for batch in tqdm(dataloader, desc="Computing embeddings"):
        images, batch_tile_ids = batch
        embeddings.append(model(images.to(device)).detach().cpu())
        tile_ids.extend(batch_tile_ids)

    embeddings = torch.cat(embeddings, dim=0).numpy()
    tile_ids = np.array(tile_ids)

    if output_path:
        np.savez_compressed(output_path,
                            embeddings=embeddings,
                            tile_ids=tile_ids)
        print(f"Embeddings saved to {output_path}")

    return embeddings, tile_ids


def load_embeddings(input_path):
    """
    Load embeddings and tile IDs from disk.
    Args:
        input_path: Path to the saved embeddings.
    Returns:
        embeddings: NumPy array of embeddings.
        tile_ids: List of corresponding tile IDs.
    """
    data = np.load(input_path)
    return data["embeddings"], data["tile_ids"]


def evaluate_on_LungHist700_paper_fold(
    model,
    data_dir="data/processed/LungHist700/",
    batch_size=128,
    num_workers=12,
    device="cuda",
    resolution="20x",
    verbose=False,
):
    train_patient_ids = [
        2, 3, 4, 5, 7, 8, 12, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 28,
        29, 30, 33, 36, 37, 38, 39, 41, 42, 45
    ]
    val_patient_ids = [1, 6, 27, 32, 44]
    test_patient_ids = [9, 13, 31, 40]
    train_patient_ids = val_patient_ids + train_patient_ids

    model = model.to(device)
    preprocess = model.get_preprocessing()

    data_dir = project_dir / data_dir
    tiles_dir = data_dir / "tiles"
    metadata = pd.read_csv(data_dir / "metadata.csv").set_index("tile_id")

    # Filter tiles by resolution
    tile_paths = [
        p for p in tiles_dir.glob("*.png")
        if metadata.loc[p.stem]["resolution"] == resolution
    ]
    tile_dataset = TileDataset(tile_paths, preprocess=preprocess)
    dataloader = DataLoader(tile_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers)

    # Compute or load embeddings
    embedding_path = project_dir / CONFIG["embedding_path"]
    if embedding_path.exists():
        embeddings, tile_ids = load_embeddings(CONFIG["embedding_path"])
    else:
        embeddings, tile_ids = compute_and_save_embeddings(
            model, dataloader, output_path=embedding_path, device=device)

    # Validate metadata presence
    missing_tiles = set(tile_ids) - set(metadata.index)
    if missing_tiles:
        raise ValueError(f"Missing metadata for tiles: {missing_tiles}")

    # Map labels and extract patient IDs
    labels = list(
        map(lambda x: CONFIG["superclass_mapping"][x],
            metadata.loc[tile_ids]["superclass"].tolist()))
    patient_ids = metadata.loc[tile_ids]["patient_id"].tolist()
    image_ids = metadata.loc[tile_ids]["original_filename"].tolist()

    df = pd.DataFrame(embeddings)
    df['image_id'] = image_ids
    df['patient_id'] = patient_ids

    # Aggregate embeddings while preserving patient_id
    averaged_df = df.groupby('image_id').agg({
        **{
            col: 'mean'
            for col in df.columns if col not in ['image_id', 'patient_id']
        },
        'patient_id':
        'first',  # Keep one patient_id (all are the same after consistency check)
    })

    # Convert back to numpy array if needed
    averaged_embeddings = averaged_df.drop(columns='patient_id').to_numpy()
    averaged_patient_ids = averaged_df['patient_id'].to_numpy()
    image_ids = averaged_df.index.to_list()

    # Ensure the shapes match
    assert len(averaged_embeddings) == len(averaged_patient_ids)
    grouped_metadata = metadata.groupby('original_filename').agg({
        'patient_id':
        lambda x: x.iloc[0] if x.nunique() == 1 else ValueError(
            f"Inconsistent patient_ids for {x.name}"),
        'superclass':
        lambda x: x.iloc[0] if x.nunique() == 1 else ValueError(
            f"Inconsistent superclass for {x.name}"),
        'label':
        lambda x: x.iloc[0] if x.nunique() == 1 else ValueError(
            f"Inconsistent superclass for {x.name}")
    })

    embeddings = averaged_embeddings
    labels = grouped_metadata.loc[image_ids]['superclass'].to_list()
    labels = list(map(lambda x: CONFIG["superclass_mapping"][x], labels))
    patient_ids = grouped_metadata.loc[image_ids]['patient_id'].to_list()

    # Split data based on patient IDs
    train_idx = [
        i for i, pid in enumerate(patient_ids) if pid in train_patient_ids
    ]
    test_idx = [
        i for i, pid in enumerate(patient_ids) if pid in test_patient_ids
    ]

    # Define pipeline and train
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('classifier', LogisticRegression(max_iter=1000))])
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
    tile_ids_test = np.array(tile_ids)[test_idx]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Display verbose information if required
    if verbose:
        print(f"Number of train tiles: {len(train_idx)}")
        print(f"Number of test tiles: {len(test_idx)}")
        print(f"Accuracy on test set: {accuracy:.4f}")

    # Return results
    return accuracy, pd.DataFrame({
        "tile_id": tile_ids_test,
        "true_label": y_test,
        "predicted_label": y_pred
    }), confusion_matrix(y_test, y_pred)


def evaluate_on_LungHist700(model,
                            n_splits=5,
                            data_dir="data/processed/LungHist700/",
                            magnification="both",
                            batch_size=128,
                            num_workers=12,
                            device="cuda",
                            verbose=False):
    model = model.to(device)
    preprocess = model.get_preprocessing()

    data_dir = project_dir / data_dir
    tiles_dir = data_dir / "tiles"
    metadata = pd.read_csv(data_dir / "metadata.csv").set_index("tile_id")

    tile_dataset = TileDataset(list(tiles_dir.glob("*.png")),
                               preprocess=preprocess)
    dataloader = DataLoader(tile_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers)

    # Compute or load embeddings
    embedding_path = project_dir / CONFIG["embedding_path"]
    if embedding_path.exists():
        embeddings, tile_ids = load_embeddings(CONFIG["embedding_path"])
    else:
        embeddings, tile_ids = compute_and_save_embeddings(
            model, dataloader, output_path=embedding_path, device=device)

    missing_tiles = set(tile_ids) - set(metadata.index)
    if missing_tiles:
        raise ValueError(f"Missing metadata for tiles: {missing_tiles}")

    labels = list(
        map(lambda x: CONFIG["superclass_mapping"][x],
            metadata.loc[tile_ids]["superclass"].tolist()))
    patient_ids = metadata.loc[tile_ids]["patient_id"].tolist()
    image_ids = metadata.loc[tile_ids]["original_filename"].tolist()

    df = pd.DataFrame(embeddings)
    df['image_id'] = image_ids
    df['patient_id'] = patient_ids

    # Check consistency of patient_id within each image_id
    def check_patient_id_consistency(group):
        unique_patient_ids = group['patient_id'].unique()
        if len(unique_patient_ids) > 1:
            raise ValueError(
                f"Inconsistent patient IDs for image_id '{group.name}': {unique_patient_ids}"
            )
        return group

    df = df.groupby('image_id',
                    group_keys=False).apply(check_patient_id_consistency)

    # Aggregate embeddings while preserving patient_id
    averaged_df = df.groupby('image_id').agg({
        **{
            col: 'mean'
            for col in df.columns if col not in ['image_id', 'patient_id']
        },
        'patient_id':
        'first',  # Keep one patient_id (all are the same after consistency check)
    })

    # Convert back to numpy array if needed
    averaged_embeddings = averaged_df.drop(columns='patient_id').to_numpy()
    averaged_patient_ids = averaged_df['patient_id'].to_numpy()
    image_ids = averaged_df.index.to_list()

    # Ensure the shapes match
    assert len(averaged_embeddings) == len(averaged_patient_ids)
    grouped_metadata = metadata.groupby('original_filename').agg({
        'patient_id':
        lambda x: x.iloc[0] if x.nunique() == 1 else ValueError(
            f"Inconsistent patient_ids for {x.name}"),
        'superclass':
        lambda x: x.iloc[0] if x.nunique() == 1 else ValueError(
            f"Inconsistent superclass for {x.name}"),
        'label':
        lambda x: x.iloc[0] if x.nunique() == 1 else ValueError(
            f"Inconsistent superclass for {x.name}")
    })

    embeddings = averaged_embeddings
    labels = grouped_metadata.loc[image_ids]['superclass'].to_list()
    labels = list(map(lambda x: CONFIG["superclass_mapping"][x], labels))
    patient_ids = grouped_metadata.loc[image_ids]['patient_id'].to_list()

    if verbose:
        print(f"Number of tiles: {len(tile_ids)}")
        print(f"Embedding dimensions: {embeddings.shape}")
        print(f"Labels distribution: {pd.Series(labels).value_counts()}")

    cv_splitter = StratifiedGroupKFold(n_splits=n_splits,
                                       shuffle=True,
                                       random_state=42)
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('classifier', LogisticRegression(max_iter=1000))])

    # Store results for analysis
    all_predictions = []
    all_true_labels = []
    all_tile_ids = []

    for fold, (train_idx, test_idx) in enumerate(
            cv_splitter.split(embeddings, labels, groups=patient_ids)):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(
            labels)[test_idx]
        tile_ids_test = np.array(tile_ids)[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Log predictions, true labels, and tile IDs
        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test)
        all_tile_ids.extend(tile_ids_test)

        if verbose:
            print(
                f"Fold {fold + 1}: Accuracy = {(y_pred == y_test).mean():.4f}")

    # Create a DataFrame for analysis
    results_df = pd.DataFrame({
        "tile_id": all_tile_ids,
        "true_label": all_true_labels,
        "predicted_label": all_predictions
    })

    # Compute overall accuracy
    accuracy = (
        results_df["true_label"] == results_df["predicted_label"]).mean()

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)

    return accuracy, results_df, conf_matrix


def get_device(gpu_id=None):
    """
    Select the appropriate device for computation.
    Args:
        gpu_id: ID of the GPU to use (None for CPU).
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        if gpu_id is not None and gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cuda:0")  # Default to the first GPU
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device


if __name__ == "__main__":
    model = BaseFeatureExtractor.get_feature_extractor(
        "UNI",
        weights_filepath=(
            "models/uni/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
            "pytorch_model.bin"),
    )

    device = get_device(gpu_id=1)
    # accuracy, results_df, conf_matrix = evaluate_on_LungHist700_paper_fold(
    accuracy, results_df, conf_matrix = evaluate_on_LungHist700(
        model,
        verbose=True,
        device=device,
    )
    print(f"Overall accuracy: {accuracy:.4f}")

    # Save results for further analysis
    results_df.to_csv("evaluation_results.csv", index=False)

    # Display confusion matrix
    conf_matrix = conf_matrix.astype(np.float64) / conf_matrix.sum(
        axis=1, keepdims=True)
    ConfusionMatrixDisplay(
        conf_matrix,
        display_labels=["nor", "aca", "scc"],
    ).plot(cmap="Blues")

    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

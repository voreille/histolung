from abc import ABC, abstractmethod
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
import torchvision.transforms as T

from histolung.models.feature_extractor import BaseFeatureExtractor
from histolung.evaluation.datasets import TileDataset

project_dir = Path(__file__).resolve().parents[2]


class BaseEvaluator(ABC):

    def __init__(self, test_loader):
        self.test_loader = test_loader

    @abstractmethod
    def evaluate(self, model):
        pass

    def plot_confusion_matrix(self,
                              y_true,
                              y_pred,
                              labels,
                              save_path="confusion_matrix.png"):
        """Plot and save a normalized confusion matrix in percentages."""
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype(np.float64) / cm.sum(
            axis=1, keepdims=True) * 100  # Normalize row-wise

        disp = ConfusionMatrixDisplay(cm_percent, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap="Blues", ax=ax, values_format=".1f")
        plt.title("Confusion Matrix (%)")
        plt.savefig(save_path)
        plt.show()


class LungHist700Evaluator(BaseEvaluator):
    """
    Evaluates SSL models on histopathology images using computed embeddings with only k-NN.
    """

    def __init__(
        self,
        superclass_mapping=None,
        data_dir="data/processed/LungHist700/",
        tile_size=224,
        batch_size=256,
        num_workers=12,
        n_splits=5,
        device="cuda",
        resolution="20x",
        preprocess=None,
    ):
        """Initialize the evaluator with only k-NN."""
        self.superclass_mapping = superclass_mapping or {
            "nor": 0,
            "aca": 1,
            "scc": 2
        }
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.resolution = resolution
        self.n_splits = n_splits

        # **Only use k-NN**
        self.knn_classifier = Pipeline([("scaler", StandardScaler()),
                                        ("classifier",
                                         KNeighborsClassifier(n_neighbors=5))])

        data_dir = project_dir / data_dir
        tiles_dir = data_dir / "tiles"
        self.metadata = pd.read_csv(data_dir /
                                    "metadata.csv").set_index("tile_id")
        if preprocess is None:
            preprocess = T.Compose([
                T.Resize((tile_size, tile_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        tile_paths = list(tiles_dir.glob("*.png"))
        if len(tile_paths) != 5636:
            raise ValueError(f"Check the integrity of the LungHist700 "
                             f"data should be 5639 tiles, {len(tile_paths)}"
                             f" tiles were found.")
        tile_dataset = TileDataset(tile_paths, transform=preprocess)
        self.dataloader = DataLoader(
            tile_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=True,
        )

    def compute_embeddings(self, model):
        """Compute embeddings dynamically for the given model."""
        model.to(self.device)
        model.eval()
        embeddings, tile_ids = [], []

        for images, batch_tile_ids in tqdm(self.dataloader,
                                           desc="Computing embeddings"):
            embeddings.append(model(images.to(self.device)).detach().cpu())
            tile_ids.extend(batch_tile_ids)

        return torch.cat(embeddings, dim=0).numpy(), np.array(tile_ids)

    def evaluate(self, embeddings, tile_ids, verbose=False):
        """
        Compute embeddings and evaluate using only k-NN.
        """
        labels = self.metadata.loc[tile_ids]["superclass"].map(
            self.superclass_mapping).to_numpy()
        patient_ids = self.metadata.loc[tile_ids]["patient_id"].to_numpy()

        # Ensure all metadata is present
        missing_tiles = set(tile_ids) - set(self.metadata.index)
        if missing_tiles:
            raise ValueError(f"Missing metadata for tiles: {missing_tiles}")

        df = pd.DataFrame(embeddings)
        df["image_id"] = self.metadata.loc[tile_ids][
            "original_filename"].to_list()
        df["patient_id"] = patient_ids

        # Aggregate embeddings while preserving patient_id
        averaged_df = df.groupby("image_id").agg({
            **{
                col: "mean"
                for col in df.columns if col not in ["image_id", "patient_id"]
            },
            "patient_id": "first",
        })

        # Convert back to numpy
        averaged_embeddings = averaged_df.drop(columns="patient_id").to_numpy()
        image_ids = averaged_df.index.to_list()

        grouped_metadata = self.metadata.groupby("original_filename").agg({
            "patient_id":
            lambda x: x.iloc[0] if x.nunique() == 1 else ValueError(
                f"Inconsistent patient_ids for {x.name}"),
            "superclass":
            lambda x: x.iloc[0] if x.nunique() == 1 else ValueError(
                f"Inconsistent superclass for {x.name}"),
        })

        embeddings = averaged_embeddings
        labels = grouped_metadata.loc[image_ids]["superclass"].map(
            self.superclass_mapping).to_numpy()
        patient_ids = grouped_metadata.loc[image_ids]["patient_id"].to_list()

        # Stratified Cross-Validation
        cv_splitter = StratifiedGroupKFold(n_splits=self.n_splits,
                                           shuffle=True,
                                           random_state=42)

        all_predictions, all_true_labels = [], []

        for fold, (train_idx, test_idx) in enumerate(
                cv_splitter.split(embeddings, labels, groups=patient_ids)):
            X_train, X_test = embeddings[train_idx], embeddings[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            self.knn_classifier.fit(X_train, y_train)
            y_pred = self.knn_classifier.predict(X_test)

            all_predictions.extend(y_pred)
            all_true_labels.extend(y_test)

            if verbose:
                print(
                    f"k-NN - Fold {fold + 1}: Accuracy = {(y_pred == y_test).mean():.4f}"
                )

        accuracy = accuracy_score(all_true_labels, all_predictions)
        conf_matrix = confusion_matrix(all_true_labels, all_predictions)

        return {
            "accuracy":
            accuracy,
            "results_df":
            pd.DataFrame({
                "image_uids": image_ids,
                "true_label": all_true_labels,
                "predicted_label": all_predictions
            }),
            "conf_matrix":
            conf_matrix
        }


def get_device(gpu_id=None):
    """Select the appropriate device for computation."""
    if torch.cuda.is_available():
        if gpu_id is not None and gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cuda:0")  # Default to first GPU
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device


def load_embeddings(input_path):
    """Load embeddings and tile IDs from disk."""
    data = np.load(input_path)
    return data["embeddings"], data["tile_ids"]


if __name__ == "__main__":
    path_embeddings = project_dir / "data/processed/LungHist700/embeddings.npz"

    if path_embeddings.exists():
        embeddings, tile_ids = load_embeddings(path_embeddings)
        evaluator = LungHist700Evaluator()
    else:
        model = BaseFeatureExtractor.get_feature_extractor(
            "UNI",
            weights_filepath=
            ("models/uni/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin"
             ),
        )
        evaluator = LungHist700Evaluator()

        device = get_device(gpu_id=1)
        embeddings, tile_ids = evaluator.compute_embeddings(model)
        # **Save embeddings for future use**
        print(f"Saving embeddings to {path_embeddings}")
        np.savez_compressed(path_embeddings,
                            embeddings=embeddings,
                            tile_ids=tile_ids)

    results = evaluator.evaluate(embeddings, tile_ids, verbose=True)

    print(f"\nk-NN Accuracy: {results['accuracy']:.4f}")

    evaluator.plot_confusion_matrix(results["results_df"]["true_label"],
                                    results["results_df"]["predicted_label"],
                                    labels=["nor", "aca", "scc"],
                                    save_path="confusion_matrix_knn.png")

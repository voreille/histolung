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


class AggregationStrategy(ABC):

    @abstractmethod
    def aggregate(self, embeddings_df):
        pass


class MeanAggregation(AggregationStrategy):

    def aggregate(self, embeddings_df):
        return embeddings_df.groupby("image_id").mean().drop(
            columns="patient_id")


class NoAggregation(AggregationStrategy):

    def aggregate(self, embeddings_df):
        return embeddings_df.drop(columns="patient_id")


class BaseEvaluator(ABC):

    def __init__(self, test_loader, aggregation_strategy: AggregationStrategy):
        """
        Initialize the evaluator.

        Args:
            test_loader (DataLoader): DataLoader for test set.
            aggregation_strategy (AggregationStrategy): Strategy for aggregating embeddings.
        """
        self.test_loader = test_loader
        self.aggregation_strategy = aggregation_strategy

    @abstractmethod
    def evaluate(self, model):
        pass

    def plot_confusion_matrix(self,
                              y_true,
                              y_pred,
                              labels,
                              save_path="confusion_matrix.png"):
        """Plot and save a normalized confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap="Blues", ax=ax, values_format=".1f")
        plt.title("Confusion Matrix (%)")
        plt.savefig(save_path)
        plt.show()


class LungHist700Evaluator(BaseEvaluator):

    def __init__(
        self,
        class_mapping=None,
        data_dir="data/processed/LungHist700/",
        tile_size=224,
        batch_size=256,
        num_workers=12,
        gpu_id=0,
        aggregation_strategy=MeanAggregation(),
        preprocess=None,
        # Default arguments for `evaluate`
        n_splits="LPO",
        seed=42,
        magnification="all",
        aggregate=True,
        n_neighbors=5,
    ):
        """
        Initialize the evaluator.

        Args:
            class_mapping (dict): Mapping of class labels to integer values.
            data_dir (str, optional): Path to the processed dataset directory.
            tile_size (int, optional): Tile image size for preprocessing.
            batch_size (int, optional): Number of samples per batch.
            num_workers (int, optional): Number of workers for DataLoader.
            gpu_id (int, optional): GPU device ID.
            aggregation_strategy (AggregationStrategy, optional): Strategy for aggregating embeddings.
            preprocess (callable, optional): Preprocessing pipeline.
            n_splits (int or str, optional): Number of splits for cross-validation ("LPO" for Leave-Patient-Out).
            seed (int, optional): Random seed for reproducibility.
            magnification (str, optional): Magnification level ("all", "20x", "40x").
        """
        device = get_device(gpu_id)

        self.class_mapping = class_mapping or {"nor": 0, "aca": 1, "scc": 2}

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.aggregation_strategy = aggregation_strategy

        # Default evaluation parameters
        self.default_n_splits = n_splits
        self.default_seed = seed
        self.default_magnification = magnification
        self.default_aggregate = aggregate

        self.knn_classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ])

        data_dir = project_dir / data_dir
        tiles_dir = data_dir / "tiles"
        self.metadata = pd.read_csv(data_dir /
                                    "metadata.csv").set_index("tile_id")

        if preprocess is None:
            preprocess = T.Compose([
                T.Resize((tile_size, tile_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        tile_paths = list(tiles_dir.glob("*.png"))
        tile_dataset = TileDataset(tile_paths, transform=preprocess)
        self.dataloader = DataLoader(tile_dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     pin_memory=True)

    def _check_n_splits(self, n_splits):
        if isinstance(n_splits, int):
            if n_splits <= 1:
                raise ValueError(
                    f"n_splits should be between 2 and {len(self.unique_patient_ids)} (number of unique patients)."
                )
        elif n_splits != "LPO":
            raise ValueError("n_splits should be an integer or 'LPO'.")

    def compute_embeddings(self, model, device=None):
        """Compute embeddings dynamically for the given model."""
        if device is None:
            device = self.device
        model.to(device)
        model.eval()
        embeddings, tile_ids = [], []

        for images, batch_tile_ids in tqdm(self.dataloader,
                                           desc="Computing embeddings"):
            embeddings.append(model(images.to(device)).detach().cpu())
            tile_ids.extend(batch_tile_ids)

        return torch.cat(embeddings, dim=0).numpy(), np.array(tile_ids)

    def filter_by_magnification(self, embeddings, tile_ids, magnification):
        """Filter embeddings and tile IDs by magnification."""
        magnification_mapping = {
            "all": ["20x", "40x"],
            "20x": ["20x"],
            "40x": ["40x"],
        }
        mask = self.metadata.loc[tile_ids]["resolution"].isin(
            magnification_mapping[magnification])
        return embeddings[mask], tile_ids[mask]

    def evaluate(self,
                 embeddings,
                 tile_ids,
                 verbose=False,
                 n_splits=None,
                 seed=None,
                 aggregate=None,
                 magnification=None,
                 n_neighbors=None):
        """Evaluate the model using k-NN classification.

        If `n_splits`, `seed`, or `magnification` are not provided, use defaults set during initialization.
        """

        # Use instance defaults if not overridden
        n_splits = n_splits if n_splits is not None else self.default_n_splits
        seed = seed if seed is not None else self.default_seed
        magnification = magnification if magnification is not None else self.default_magnification
        aggregate = aggregate if aggregate is not None else self.default_aggregate
        if n_neighbors is None:
            knn_classifier = self.knn_classifier
        else:
            knn_classifier = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors)),
            ])

        self._check_n_splits(n_splits)

        if magnification not in ["all", "20x", "40x"]:
            raise ValueError(
                "Invalid magnification. Choose from 'all', '20x', '40x'")

        if magnification != "all":
            embeddings, tile_ids = self.filter_by_magnification(
                embeddings, tile_ids, magnification)

        labels = self.metadata.loc[tile_ids]["superclass"].map(
            self.class_mapping).to_numpy()
        patient_ids = self.metadata.loc[tile_ids]["patient_id"].to_numpy()

        if aggregate:
            df = pd.DataFrame(embeddings)
            df["image_id"] = self.metadata.loc[tile_ids][
                "original_filename"].to_list()
            df["patient_id"] = patient_ids

            aggregated_df = self.aggregation_strategy.aggregate(df)
            image_ids = aggregated_df.index.to_list()

            grouped_metadata = self.metadata.groupby("original_filename").agg({
                "patient_id":
                "first",
                "superclass":
                "first"
            })

            embeddings = aggregated_df.to_numpy()
            labels = grouped_metadata.loc[image_ids]["superclass"].map(
                self.class_mapping).to_numpy()
            patient_ids = grouped_metadata.loc[image_ids]["patient_id"].values
        else:
            image_ids = tile_ids

        # Create splits
        if n_splits == "LPO":
            splits = [(patient_ids != pid, patient_ids == pid)
                      for pid in np.unique(patient_ids)]
        else:
            cv_splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            splits = list(
                cv_splitter.split(
                    embeddings,
                    labels,
                    groups=patient_ids,
                ))

        all_predictions, all_true_labels = [], []
        accuracies = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            assert set(patient_ids[train_idx]).isdisjoint(
                set(patient_ids[test_idx]))
            X_train, X_test = embeddings[train_idx], embeddings[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            knn_classifier.fit(X_train, y_train)
            y_pred = knn_classifier.predict(X_test)

            all_predictions.extend(y_pred)
            all_true_labels.extend(y_test)
            fold_accuracy = (y_pred == y_test).mean()
            accuracies.append(fold_accuracy)

            if verbose:
                print(
                    f"k-NN - Fold {fold + 1}: Accuracy = {fold_accuracy:.4f}")

        concatenated_accuracy = accuracy_score(all_true_labels,
                                               all_predictions)
        conf_matrix = confusion_matrix(all_true_labels, all_predictions)

        return {
            "concatenated_accuracy":
            concatenated_accuracy,
            "mean_accuracy":
            np.mean(accuracies),
            "std_accuracy":
            np.std(accuracies),
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

    path_embeddings = project_dir / f"data/processed/LungHist700_20x/embeddings.npz"
    data_dir = project_dir / f"data/processed/LungHist700_20x"

    if path_embeddings.exists():
        embeddings, tile_ids = load_embeddings(path_embeddings)
        evaluator = LungHist700Evaluator(data_dir=data_dir)
    else:
        device = get_device(gpu_id=1)
        model = BaseFeatureExtractor.get_feature_extractor(
            "UNI",
            weights_filepath=
            ("models/uni/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin"
             ))
        evaluator = LungHist700Evaluator(gpu_id=1, data_dir=data_dir)

        embeddings, tile_ids = evaluator.compute_embeddings(model)
        # **Save embeddings for future use**
        print(f"Saving embeddings to {path_embeddings}")
        np.savez_compressed(path_embeddings,
                            embeddings=embeddings,
                            tile_ids=tile_ids)

    results = evaluator.evaluate(
        embeddings,
        tile_ids,
        verbose=True,
        #  n_splits=5,
        magnification="20x")

    print(
        f"\nk-NN Concatenated Accuracy: {results['concatenated_accuracy']:.4f}"
    )
    print(
        f"\nk-NN Mean Accuracy \u00B1 2*STD : {results['mean_accuracy']:.4f} \u00B1 {2*results['std_accuracy']:.4f}"
    )

    evaluator.plot_confusion_matrix(results["results_df"]["true_label"],
                                    results["results_df"]["predicted_label"],
                                    labels=["nor", "aca", "scc"],
                                    save_path="confusion_matrix_knn.png")

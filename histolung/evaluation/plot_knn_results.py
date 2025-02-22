import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from histolung.evaluation.evaluators import LungHist700Evaluator
from histolung.models.models_darya import MoCoV2Encoder


def get_device(gpu_id=None):
    """Select the appropriate device for computation."""
    if torch.cuda.is_available():
        if gpu_id is not None and gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda:0")  # Default to first GPU
    else:
        device = torch.device("cpu")
    print(f"Using {device}")
    return device


def load_model(checkpoint_path, device):
    """Load the MoCoV2 model from a given checkpoint."""
    if not checkpoint_path.exists():
        print(f"Checkpoint {checkpoint_path} not found, skipping...")
        return None

    model = MoCoV2Encoder()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.encoder_q.to(device).eval()


def evaluate_checkpoint(evaluator, checkpoint_path, device):
    """Evaluate a model for a specific epoch and return the results."""
    epoch = int(checkpoint_path.stem.split("_")[-1])
    model = load_model(checkpoint_path, device)

    if model is None:
        return None  # Skip if checkpoint does not exist

    # Compute embeddings
    embeddings, tile_ids = evaluator.compute_embeddings(model)

    # Evaluate embeddings
    results = evaluator.evaluate(
        embeddings,
        tile_ids,
        verbose=True,
    )

    return {
        "epoch": epoch,
        "concatenated_accuracy": results["concatenated_accuracy"],
        "mean_accuracy": results["mean_accuracy"],
        "std_accuracy": results["std_accuracy"],
    }


def plot_results(results_per_epoch, save_path):
    """Plot accuracy evolution across epochs and save the figure."""
    if not results_per_epoch:
        print("No valid results to plot.")
        return

    epochs = [res["epoch"] for res in results_per_epoch]
    concatenated_acc = [
        res["concatenated_accuracy"] for res in results_per_epoch
    ]
    mean_acc = [res["mean_accuracy"] for res in results_per_epoch]
    std_acc = [res["std_accuracy"] for res in results_per_epoch]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs,
             concatenated_acc,
             label="Concatenated Accuracy",
             marker="o")
    plt.errorbar(epochs,
                 mean_acc,
                 yerr=std_acc,
                 label="Mean Accuracy ± STD",
                 fmt="o-",
                 capsize=5)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MoCo ResNet50 Evaluation Over Epochs")
    plt.legend()
    plt.grid()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


def main():
    # Paths
    n_splits = 5
    magnification = "all"
    tiling_magnification = "10x"
    aggregate = False
    exp_name = "superpixels_moco_org"
    project_dir = Path(__file__).parents[2].resolve()
    checkpoint_dir = Path(
        "/mnt/nas7/data/Personal/Darya/saved_models") / exp_name
    plot_filename = f"evaluation_plot_exp_{exp_name}__n_splits_{n_splits}__mag_{magnification}__tilemag_{tiling_magnification}"
    if not aggregate:
        plot_filename += "__no_agg"

    plot_save_path = project_dir / f"reports/{exp_name}/{plot_filename}.png"

    # Device setup
    device = get_device(gpu_id=0)

    # Initialize evaluator
    evaluator = LungHist700Evaluator(
        batch_size=256,
        num_workers=4,
        gpu_id=0,
        data_dir=project_dir /
        f"data/processed/LungHist700_{tiling_magnification}",
        n_splits=5,
        aggregate=aggregate,
    )

    checkpoint_paths = [f for f in checkpoint_dir.glob("*.pth")]

    # Evaluate all epochs
    results_per_epoch = [
        result for checkpoint_path in checkpoint_paths
        if (result := evaluate_checkpoint(evaluator, checkpoint_path, device)
            ) is not None
    ]

    # Plot and save results
    plot_results(results_per_epoch, plot_save_path)


if __name__ == "__main__":
    main()

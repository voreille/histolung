from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import openslide
import cv2 as cv
import seaborn as sns
import scipy.ndimage as ndimage
from natsort import natsorted

from histolung.models.models import PretrainedModelLoader, MILModel
from histolung.utils import yaml_load
from histolung.legacy.datasets import Dataset_instance_MIL


def smooth_heatmap(heatmap, sigma):
    """Applies Gaussian smoothing to the heatmap."""
    heatmap_smooth = ndimage.gaussian_filter(heatmap, sigma=sigma, order=0)
    return np.array(heatmap_smooth)


def predict(net, data_loader, device):
    """Performs a forward pass over the data_loader and returns predictions and attention weights."""
    net.eval()

    all_attention_weights = []
    all_preds = []

    with torch.no_grad():
        for batch_data in data_loader:
            # Move batch to device
            batch_data = batch_data.to(device, non_blocking=True)

            # Forward pass through the network
            preds, attention_weights = net(batch_data)

            # Convert predictions and attention weights to CPU numpy arrays
            preds_np = preds.cpu().numpy()
            attention_weights_np = attention_weights.cpu().numpy()

            all_preds.extend(preds_np)
            all_attention_weights.extend(attention_weights_np)

    # Convert to numpy arrays for final processing
    all_preds = np.array(all_preds)
    all_attention_weights = np.array(all_attention_weights)

    return all_preds, all_attention_weights


@click.command()
@click.option(
    "--wsi_name",
    default="TCGA-18-3417-01Z-00-DX1.tif",
    # prompt="Name of the WSI to perform study",
    help="Name of the WSI to perform study",
)
@click.option(
    "--sigma",
    default=8,
    # prompt="Value of sigma applied to the Gaussian filter",
    help="Value of sigma applied to the Gaussian filter",
)
def main(wsi_name, sigma):
    # Set up paths
    base_path = Path(__file__).resolve().parents[2]
    data_dir = base_path / "data" / "tcga_old"
    wsi_dir = data_dir / "wsi"
    patch_dir = data_dir / "patches"
    mask_dir = data_dir / "mask"
    model_dir = base_path / "models" / "MIL" / "f_MIL_res34v2_v2_rumc_best_cosine_v3"

    output_dir = base_path / "reports" / "heatmaps" / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    device = set_device_and_seed(device=0, seed=33)

    # Load labels and get ground truth
    labels_path = data_dir / "labels_tcga_all.csv"
    groundtruth = get_groundtruth_label(wsi_name, labels_path)

    # Load model configuration and checkpoint
    cfg_path = model_dir / "config_f_MIL_res34v2_v2_rumc_best_cosine_v3.yml"
    checkpoint_path = model_dir / "fold_0" / "checkpoint.pt"
    cfg = yaml_load(cfg_path)
    net = load_model(cfg, checkpoint_path, device)

    # Prepare data transformations
    preprocess = prepare_data_transformations(
        cfg,
        net.model_loader.resize_param,
    )

    # Open WSI and read mask
    wsi_path = wsi_dir / wsi_name
    mask_path = mask_dir / wsi_name / f"{wsi_name}_mask_use.png"
    wsi_file, mask, thumbnail = open_wsi_and_mask(wsi_path, mask_path)

    # Read patches and metadata
    sample_dir = patch_dir / wsi_name
    metadata_file = sample_dir / f"{wsi_name}_coords_densely.csv"
    patches, coords_x, coords_y, names = read_patches_and_metadata(
        sample_dir, metadata_file)

    # Set up DataLoader
    batch_size_instance = 1
    num_workers = 2
    data_loader = DataLoader(
        Dataset_instance_MIL(patches, preprocess=preprocess),
        batch_size=batch_size_instance,
        num_workers=num_workers,
    )

    pred_scores, attentions = net.process_dataloader(data_loader)

    print("Predictions for classes: SCLC, LUAD, LUSC, Normal")
    print(f"Prediction scores: {pred_scores}")

    final_prediction_id = pred_scores.argmax()
    prediction_labels = ["SCLC", "LUAD", "LUSC", "Normal"]
    final_prediction = prediction_labels[final_prediction_id]

    # Create heatmap
    mask_shape = mask.shape
    heatmap = create_heatmap(mask_shape, attentions, coords_x, coords_y,
                             final_prediction, sigma)

    # Save results
    save_results(wsi_name, final_prediction, groundtruth, output_dir)

    # Plot and save heatmap
    cmap = get_colormap(final_prediction)
    plot_and_save_heatmap(thumbnail, heatmap, cmap,
                          output_dir / f"heatmap_{wsi_name[:-4]}.png")

    print(f"Heatmap saved in {output_dir}")


def set_device_and_seed(device=0, seed=33):
    """Sets the device and seed for reproducibility."""
    device = torch.device(
        f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    return device


def get_groundtruth_label(wsi_name, labels_path):
    """Retrieves the ground truth label for the given WSI."""
    labels_df = pd.read_csv(labels_path, index_col=0)
    wsi_id = wsi_name.split(".")[0]
    row = labels_df.loc[wsi_id]

    if row["cancer_nscc_adeno"] == 1:
        groundtruth = "LUAD"
    elif row["cancer_nscc_squamous"] == 1:
        groundtruth = "LUSC"
    else:
        groundtruth = "Unknown"

    print(
        f"Processing sample: {wsi_name} with Ground truth label: {groundtruth}"
    )
    return groundtruth


def load_model(cfg, checkpoint_path, device):
    """Loads the model and its weights."""
    print(f"Loading Model using {cfg.model.model_name} as backbone")

    model_loader = PretrainedModelLoader(
        cfg.model.model_name,
        cfg.model.num_classes,
        freeze=cfg.model.freeze_weights,
        num_freezed_layers=cfg.model.num_frozen_layers,
        dropout=cfg.model.dropout,
        embedding_bool=cfg.model.embedding_bool,
        pool_algorithm=cfg.model.pool_algorithm,
    )

    hidden_space_len = cfg.model.hidden_space_len

    net = MILModel(model_loader, hidden_space_len, cfg)

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint["model_state_dict"], strict=False)
    net.to(device)
    net.eval()

    return net


def prepare_data_transformations(cfg, resize_param):
    """Prepares data transformations for preprocessing."""
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.stddev),
        transforms.Resize(size=(resize_param, resize_param), antialias=True),
    ])
    return preprocess


def open_wsi_and_mask(wsi_path, mask_path):
    """Opens the WSI file and reads the mask."""
    wsi_file = openslide.open_slide(str(wsi_path))

    mask = cv.imread(str(mask_path))
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)

    print(f"Mask shape: {mask.shape}")

    thumbnail = wsi_file.get_thumbnail((mask.shape[1], mask.shape[0]))

    return wsi_file, mask, thumbnail


def read_patches_and_metadata(sample_dir, metadata_file):
    """Reads the patches and their metadata."""
    metadata_preds = pd.read_csv(metadata_file, header=None)

    names = metadata_preds.iloc[:, 0].values
    coords_x = metadata_preds.iloc[:, 3].values
    coords_y = metadata_preds.iloc[:, 2].values

    patches = natsorted([[str(i)] for i in sample_dir.rglob("*.png")], key=str)

    return patches, coords_x, coords_y, names


def extract_features(net,
                     data_loader,
                     coords_x,
                     coords_y,
                     names,
                     device,
                     downsample_factor=32):
    """Extracts features from patches using the model."""
    n_elems = len(names)
    dicts = []
    features = []
    with torch.no_grad():
        for i, patch in enumerate(data_loader):
            patch = patch.to(device, non_blocking=True)

            coord_x = int(coords_x[i] / downsample_factor)
            coord_y = int(coords_y[i] / downsample_factor)

            # Extract features
            feats = net.conv_layers(patch)
            feats = feats.view(-1, net.fc_input_features)
            feats_np = feats.cpu().numpy()

            features.extend(feats_np)

            d = {
                "ID": names[i],
                "coord_x": coord_x,
                "coord_y": coord_y,
                "prob": None,
            }
            dicts.append(d)

    features_np = np.array(features).reshape(n_elems, -1)
    return features_np, dicts


def get_predictions(net, features, device):
    """Gets predictions and attention weights from the model."""
    inputs = torch.tensor(features).float().to(device, non_blocking=True)
    with torch.no_grad():
        pred_wsi, attention_weights = net.forward_precomputed(inputs)

    pred_wsi = pred_wsi.cpu().numpy()
    attention_weights = attention_weights.cpu().numpy()

    print("Predictions for classes: SCLC, LUAD, LUSC, Normal")
    print(f"Prediction scores: {pred_wsi}")

    final_prediction_id = pred_wsi.argmax()
    prediction_labels = ["SCLC", "LUAD", "LUSC", "Normal"]
    final_prediction = prediction_labels[final_prediction_id]

    return final_prediction, pred_wsi, attention_weights


def create_heatmap(mask_shape,
                   attentions,
                   coords_x,
                   coords_y,
                   final_prediction,
                   sigma,
                   downsample_factor=32):
    """Creates a heatmap based on attention weights."""
    mask_empty = np.zeros((mask_shape[0], mask_shape[1]))
    patch_size = int(224 / downsample_factor)

    attentions = attentions.squeeze()  # Shape: (num_classes, num_instances)
    class_index = ["SCLC", "LUAD", "LUSC", "Normal"].index(final_prediction)
    attentions_for_class = attentions[class_index, :]

    for i in range(len(coords_x)):
        x_cord_m = int(coords_x[i] / downsample_factor)
        y_cord_m = int(coords_y[i] / downsample_factor)
        x_cord_f = x_cord_m + patch_size
        y_cord_f = y_cord_m + patch_size
        prob = attentions_for_class[i]

        mask_empty[x_cord_m:x_cord_f, y_cord_m:y_cord_f] = prob

    heatmap_np = np.uint8(mask_empty * 600)
    heatmap_smooth_np = smooth_heatmap(heatmap_np, sigma)

    return heatmap_smooth_np


def save_results(wsi_name, final_prediction, groundtruth, output_dir):
    """Saves the prediction results to a CSV file."""
    df_prediction = pd.DataFrame({
        "filename": [wsi_name],
        "prediction": [final_prediction],
        "groundtruth": [groundtruth],
    })

    filename_prediction = output_dir / "predictions.csv"
    df_prediction.to_csv(filename_prediction, index=False)

    print(f"=== Final prediction of the model: {final_prediction} ===")
    print(f"=== Ground truth: {groundtruth} ===")


def get_colormap(final_prediction):
    """Returns the colormap corresponding to the final prediction."""
    colormap_dict = {
        "SCLC": sns.color_palette("YlOrBr", 255, as_cmap=True),
        "LUAD": sns.color_palette("Greens", 255, as_cmap=True),
        "LUSC": sns.color_palette("Reds", 255, as_cmap=True),
        "Normal": sns.color_palette("Blues", 255, as_cmap=True),
    }
    return colormap_dict.get(final_prediction,
                             sns.color_palette("Greys", 255, as_cmap=True))


def plot_and_save_heatmap(thumbnail, heatmap, cmap, output_path):
    """Plots and saves the heatmap overlayed on the thumbnail."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(thumbnail)
    ax.imshow(15 * heatmap, alpha=0.7, cmap=cmap)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()

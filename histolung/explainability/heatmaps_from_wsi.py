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
from histolung.data.tile_image import tile_image
from histolung.data.utils_pyhist import compute_xy_coordinates, binarize_mask

base_path = Path(__file__).resolve().parents[2]
default_wsi_path = str(base_path /
                       "data/tcga_old/wsi/TCGA-18-3417-01Z-00-DX1.tif")


def smooth_heatmap(heatmap, sigma):
    """Applies Gaussian smoothing to the heatmap."""
    heatmap_smooth = ndimage.gaussian_filter(heatmap, sigma=sigma, order=0)
    return np.array(heatmap_smooth)


def available_magnifications(mpp, level_downsamples):
    mpp = float(mpp)
    if (mpp < 0.26):
        magnification = 40
    else:
        magnification = 20

    mags = []
    for level in level_downsamples:
        mags.append(magnification / level)

    return mags


def filter_patches(wsi_path, pyhist_out_dir):
    wsi_name = pyhist_out_dir.name
    binary_mask = cv.imread(str(pyhist_out_dir / f"segmented_{wsi_name}.png"))
    binary_mask[binary_mask == 255] = 1
    mask_shape = binary_mask.shape
    binary_mask = cv.resize(
        binary_mask, (int(mask_shape[1] * 0.5), int(mask_shape[0] * 0.5)))
    mask_shape = binary_mask.shape

    slide = openslide.OpenSlide(
        str(Path(datadir / filename.parent.stem / f"{filename.stem}.tif")))
    thumbnail = slide.get_thumbnail((mask_shape[1], mask_shape[0]))

    thumbnail_data = np.array(thumbnail)
    if thumbnail_data.shape != mask_shape:
        thumbnail_data = cv.resize(thumbnail_data,
                                   (mask_shape[1], mask_shape[0]))

    lower, upper = eval_histogram_threshold(binary_mask, thumbnail_data)
    print(
        f"Set an lower threshold of {lower} and upper {upper} to compute the histogram"
    )

    patches_metadata = pd.read_csv(Path(filename / "tile_selection.tsv"),
                                   sep='\t').set_index("Tile")

    patches_path = [i for i in filename.rglob("*.png") if "tiles" in str(i)]

    patch = cv.imread(str(patches_path[0]))
    patch_shape = patch.shape
    total_pixels_patch = patch_shape[0] * patch_shape[1]
    filtered_patches = []
    names = []
    all_row = []
    all_col = []

    for image_patch in tqdm(patches_path,
                            desc=f"Filtering patches of {filename.stem}"):

        image = cv.imread(str(image_patch))
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        histo = get_histogram(gray_image, lower, upper)

        total_pixels_in_range = np.sum(histo)

        if (total_pixels_in_range > 0.6 * total_pixels_patch):
            name = image_patch.stem
            names.append(name)
            all_row.append(patches_metadata.loc[name]['Row'])
            all_col.append(patches_metadata.loc[name]['Column'])
            filtered_patches.append(image_patch)

    # Create .csv with metadata information of the filtered patches
    outputdir_metadata = Path(filename /
                              f"{filename.stem}_densely_filtered_metadata.csv")

    file_metadata = {'patch_name': names, 'row': all_row, 'column': all_col}
    df_metadata = pd.DataFrame.from_dict(file_metadata)
    df_metadata.sort_values(by='patch_name',
                            axis=0,
                            ascending=True,
                            inplace=True)
    df_metadata.to_csv(outputdir_metadata, index=False)

    # Create .csv with filtered parches path
    outputdir_paths = Path(filename /
                           f"{filename.stem}_densely_filtered_paths.csv")
    file_path = {'filtered_patch_path': filtered_patches}
    df_paths = pd.DataFrame.from_dict(file_path)
    df_paths.sort_values(by='filtered_patch_path',
                         axis=0,
                         ascending=True,
                         inplace=True)
    df_paths.to_csv(outputdir_paths, index=False)

    print(
        f"Filtered patches: {len(filtered_patches)} from a total of {len(patches_path)}"
    )
    print(
        f"Filtered .csv for {filename.stem} saved on {outputdir_paths.parent}")


@click.command()
@click.option(
    "--wsi_path",
    default=default_wsi_path,
    # prompt="Name of the WSI to perform study",
    help="Name of the WSI to perform study",
)
@click.option(
    "--sigma",
    default=8,
    # prompt="Value of sigma applied to the Gaussian filter",
    help="Value of sigma applied to the Gaussian filter",
)
def main(wsi_path, sigma):
    # Set up paths
    wsi_name = Path(wsi_path).name.split(".")[0]

    # tile the image
    pyhist_output_dir = base_path / "data/test_heatmaps"

    sample_dir = pyhist_output_dir / wsi_name / f"{wsi_name}_tiles"

    # slide = openslide.open_slide(str(wsi_path))

    # mpp = slide.properties["openslide.mpp-x"]

    # level_downsamples = slide.level_downsamples
    # mags = available_magnifications(mpp, level_downsamples)
    # if mags[0] == 40:
    #     downsample = 2
    # elif mags[0] == 20:
    #     downsample = 1

    # TODO: Fix magnification, here it's 20
    downsample = 1

    # hardcoded from Lluis's code..
    downsample_factor = 32

    if not sample_dir.exists():
        tile_image(
            wsi_path,
            pyhist_output_dir,
            downsample=downsample,
            mask_downsample=downsample_factor,
        )
        mask_path = binarize_mask(pyhist_output_dir / wsi_name /
                                  f"segmented_{wsi_name}.ppm")

        filter_patches(pyhist_output_dir)
    else:
        print("output directory already exists, so tiles are not recomputed")

    mask_path = binarize_mask(pyhist_output_dir / wsi_name /
                              f"segmented_{wsi_name}.ppm")

    tile_selection_df = pd.read_csv(
        pyhist_output_dir / wsi_name / "tile_selection.tsv",
        sep="\t",
    )
    tile_selection_df = compute_xy_coordinates(tile_selection_df)

    # mask_dir = pyhist_output_dir / wsi_name
    model_dir = base_path / "models" / "MIL" / "f_MIL_res34v2_v2_rumc_best_cosine_v3"

    # get all the patches path
    patches = [f for f in sample_dir.glob("*.png")]
    patches_selected = [f.name.split(".")[0] for f in patches]

    # format for dataloader
    patches = [[str(f)] for f in patches]

    tile_selection_df = tile_selection_df[tile_selection_df['Tile'].isin(
        patches_selected)]

    # TODO: Fix convention coordinate, here we invert x and y
    coords_x = tile_selection_df["coord_y"].values
    coords_y = tile_selection_df["coord_x"].values

    output_dir = base_path / "reports" / "heatmaps" / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    device = set_device_and_seed(device=0, seed=33)

    # Load labels and get ground truth

    # TODO: change this hardcoded path
    labels_path = base_path / "data/tcga_old/labels_tcga_all.csv"
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
    # mask_path = mask_dir / wsi_name / f"{wsi_name}_mask_use.png"
    wsi_file, mask, thumbnail = open_wsi_and_mask(wsi_path, mask_path)

    # Read patches and metadata
    # metadata_file = sample_dir / f"{wsi_name}_coords_densely.csv"
    # patches, coords_x, coords_y, names = read_patches_and_metadata(
    #     sample_dir, metadata_file)

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
    heatmap = create_heatmap(
        mask_shape,
        attentions,
        coords_x,
        coords_y,
        final_prediction,
        sigma,
        downsample_factor=downsample_factor,
    )

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

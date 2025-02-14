from pathlib import Path
from PIL import Image
import openslide
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Constants
SEGMENTS_DIR = Path("/home/valentin/workspaces/histolung/data/interim/superpixels")
CPTAC_PATH = Path("/mnt/nas6/data/CPTAC")
TCGA_PATH = Path("/mnt/nas7/data/TCGA_Lung_svs")
TARGET_AVERAGE_AREA = 672**2  # µm²
REFERENCE_MAGNIFICATION = 40  # Reference magnification for normalization

def get_wsi_path(wsi_id):
    """Find the corresponding WSI file in CPTAC or TCGA."""
    for dataset_path in [CPTAC_PATH, TCGA_PATH]:
        wsi_file = next(dataset_path.rglob(f"{wsi_id}*.svs"), None)
        if wsi_file:
            return wsi_file
    raise FileNotFoundError(f"No WSI found for {wsi_id} in CPTAC or TCGA")

def compute_scale_factor(wsi_image, mask_size):
    """Compute the scaling factor between WSI and mask dimensions."""
    wsi_width, wsi_height = wsi_image.dimensions
    mask_width, mask_height = mask_size
    return (wsi_width / mask_width, wsi_height / mask_height)

def get_mpp(wsi_image):
    """Retrieve the microns per pixel (MPP) value from OpenSlide properties."""
    mpp_x = float(wsi_image.properties.get(openslide.PROPERTY_NAME_MPP_X, 1))  # Default 1µm if missing
    mpp_y = float(wsi_image.properties.get(openslide.PROPERTY_NAME_MPP_Y, mpp_x))  # Default to mpp_x if missing
    return mpp_x, mpp_y

def get_magnification(wsi_image):
    """Retrieve the objective magnification of the WSI."""
    mag = wsi_image.properties.get("aperio.AppMag", None) or wsi_image.properties.get("openslide.objective-power", None)
    return float(mag) if mag else None

def process_superpixel_mask(superp_segments_path):
    """Process a single superpixel segmentation mask and compute mean area per WSI."""
    try:
        superp_mask = Image.open(superp_segments_path)
        wsi_id = superp_segments_path.stem.replace("_segments", "")

        # Find corresponding WSI
        wsi_image = openslide.OpenSlide(str(get_wsi_path(wsi_id)))

        # Compute scale factor, MPP, and magnification
        scale_x, scale_y = compute_scale_factor(wsi_image, superp_mask.size)
        mpp_x, mpp_y = get_mpp(wsi_image)
        magnification = get_magnification(wsi_image)

        # Normalize pixel size to 40x
        if magnification and magnification != REFERENCE_MAGNIFICATION:
            norm_mpp_x = mpp_x * (magnification / REFERENCE_MAGNIFICATION)
            norm_mpp_y = mpp_y * (magnification / REFERENCE_MAGNIFICATION)
        else:
            norm_mpp_x, norm_mpp_y = mpp_x, mpp_y

        # Convert segmentation mask to NumPy array and filter labels
        np_mask = np.array(superp_mask)
        labels = np.unique(np_mask)
        labels = labels[labels != 0]  # Remove background label (0)

        # Compute superpixel areas
        areas_pixels = np.bincount(np_mask.flatten())[labels]  # Efficient area computation

        # Convert areas from pixels² to µm²
        pixel_to_micron = (mpp_x * scale_x) * (mpp_y * scale_y)
        areas_microns = areas_pixels * pixel_to_micron

        # Compute mean superpixel area for this WSI
        mean_area = np.mean(areas_microns) if len(areas_microns) > 0 else None
        return wsi_id, mean_area, magnification, mpp_x, mpp_y, norm_mpp_x, norm_mpp_y

    except FileNotFoundError as e:
        print(f"Skipping {superp_segments_path.name}: {e}")
        return None, None, None, None, None, None, None

def process_all_superpixels(segments_dir, num_workers=8):
    """Process all segmentation masks in parallel."""
    segment_paths = list(segments_dir.rglob("*.tiff"))

    wsi_areas = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(executor.map(process_superpixel_mask, segment_paths), total=len(segment_paths))
        )

    # Filter out None values
    wsi_areas = [result for result in results if result[0] is not None]

    # Convert to DataFrame
    df = pd.DataFrame(wsi_areas, columns=["wsi_id", "mean_superpixel_area", "magnification", "mpp_x", "mpp_y", "norm_mpp_x", "norm_mpp_y"])
    return df

def compute_statistics(df):
    """Compute overall statistics and save results."""
    df = df.dropna()  # Remove WSIs with no valid superpixels
    overall_mean = df["mean_superpixel_area"].mean()
    overall_median = df["mean_superpixel_area"].median()
    deviation_percent = abs(overall_mean - TARGET_AVERAGE_AREA) / TARGET_AVERAGE_AREA * 100.0

    print("\n=== Superpixel Segmentation QC Per WSI ===")
    print(f"Total WSIs Processed: {len(df)}")
    print(f"Overall Median Superpixel Area: {overall_median:.2f} µm²")
    print(f"Overall Mean Superpixel Area: {overall_mean:.2f} µm²")
    print(f"Target Superpixel Area: {TARGET_AVERAGE_AREA} µm²")
    print(f"Deviation from target: {deviation_percent:.2f}%")

    # Save results to CSV
    df.to_csv("superpixel_qc_per_wsi.csv", index=False)
    print("\nSaved results to superpixel_qc_per_wsi.csv")

if __name__ == "__main__":
    df_superpixels = process_all_superpixels(SEGMENTS_DIR, num_workers=24)  # Adjust workers as needed
    compute_statistics(df_superpixels)

from pathlib import Path
from PIL import Image
import openslide
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Constants
SEGMENTS_DIR = Path(
    "/home/valentin/workspaces/histolung/data/interim/superpixels")
CPTAC_PATH = Path("/mnt/nas6/data/CPTAC")
TCGA_PATH = Path("/mnt/nas7/data/TCGA_Lung_svs")
TARGET_AVERAGE_AREA = 672**2  # µm²


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
    return ((wsi_width / mask_width) + (wsi_height / mask_height)) / 2


def get_mpp(wsi_image):
    """Retrieve the microns per pixel (MPP) value from OpenSlide properties."""
    return float(wsi_image.properties.get(openslide.PROPERTY_NAME_MPP_X,
                                          1))  # Default 1µm if missing


def process_superpixel_mask(superp_segments_path):
    """Process a single superpixel segmentation mask to compute areas."""
    try:
        superp_mask = Image.open(superp_segments_path)
        wsi_id = superp_segments_path.stem.replace("_segments", "")

        # Find corresponding WSI
        wsi_image = openslide.OpenSlide(str(get_wsi_path(wsi_id)))

        # Compute scale factor and MPP
        scale_factor = compute_scale_factor(wsi_image, superp_mask.size)
        mpp_x = get_mpp(wsi_image)

        # Convert segmentation mask to NumPy array and filter labels
        np_mask = np.array(superp_mask)
        labels = np.unique(np_mask)
        labels = labels[labels != 0]  # Remove background label (0)

        # Compute superpixel areas
        areas_pixels = np.bincount(
            np_mask.flatten())[labels]  # Efficient area computation

        # Convert areas from pixels² to µm²
        pixel_to_micron = (mpp_x * scale_factor)**2
        areas_microns = areas_pixels * pixel_to_micron

        return areas_microns.tolist()

    except FileNotFoundError as e:
        print(f"Skipping {superp_segments_path.name}: {e}")
        return []


def process_all_superpixels(segments_dir, num_workers=8):
    """Process all segmentation masks in parallel."""
    segment_paths = list(segments_dir.rglob("*.tiff"))

    all_areas = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(executor.map(process_superpixel_mask, segment_paths),
                 total=len(segment_paths)))

    # Flatten the list of lists
    for areas in results:
        all_areas.extend(areas)

    return all_areas


def compute_statistics(areas):
    """Compute median, mean, and deviation from target."""
    median_area = np.median(areas)
    average_area = np.mean(areas)
    deviation_percent = abs(average_area -
                            TARGET_AVERAGE_AREA) / TARGET_AVERAGE_AREA * 100.0

    print("\n=== Superpixel Segmentation Statistics ===")
    print(f"Total Superpixels Processed: {len(areas)}")
    print(f"Median Superpixel Area: {median_area:.2f} µm²")
    print(f"Average Superpixel Area: {average_area:.2f} µm²")
    print(f"Target Superpixel Area: {TARGET_AVERAGE_AREA} µm²")
    print(f"Deviation from target: {deviation_percent:.2f}%")


if __name__ == "__main__":
    all_superpixel_areas = process_all_superpixels(
        SEGMENTS_DIR, num_workers=24)  # Adjust workers as needed
    compute_statistics(all_superpixel_areas)

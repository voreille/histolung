from pathlib import Path
from PIL import Image
import openslide
import numpy as np

# Constants
SEGMENTS_DIR = Path(
    "/home/valentin/workspaces/histolung/data/interim/debug_superpixels/segments/"
)
CPTAC_PATH = Path("/mnt/nas6/data/CPTAC")
TARGET_AVERAGE_AREA = 672**2  # µm²


def get_wsi_path(wsi_id):
    """Find the corresponding WSI file."""
    wsi_file = next(CPTAC_PATH.rglob(f"{wsi_id}*.svs"), None)
    if not wsi_file:
        raise FileNotFoundError(f"No WSI found for {wsi_id}")
    return wsi_file


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
    areas_pixels = [np.sum(np_mask == label) for label in labels]

    # Convert areas from pixels² to µm²
    pixel_to_micron = (mpp_x * scale_factor)**2
    areas_microns = [area * pixel_to_micron for area in areas_pixels]

    return areas_microns


def process_all_superpixels(segments_dir):
    """Process all segmentation masks in the given directory."""
    all_areas = []

    for superp_segments_path in segments_dir.glob("*.tiff"):
        try:
            areas = process_superpixel_mask(superp_segments_path)
            all_areas.extend(areas)
        except FileNotFoundError as e:
            print(f"Skipping {superp_segments_path.name}: {e}")

    return all_areas


def compute_statistics(areas):
    """Compute median, mean, and deviation from target."""
    median_area = np.median(areas)
    average_area = np.mean(areas)
    deviation_percent = abs(average_area -
                            TARGET_AVERAGE_AREA) / TARGET_AVERAGE_AREA * 100.0

    print(f"\n=== Superpixel Segmentation Statistics ===")
    print(f"Total Superpixels Processed: {len(areas)}")
    print(f"Median Superpixel Area: {median_area:.2f} µm²")
    print(f"Average Superpixel Area: {average_area:.2f} µm²")
    print(f"Target Superpixel Area: {TARGET_AVERAGE_AREA} µm²")
    print(f"Deviation from target: {deviation_percent:.2f}%")


if __name__ == "__main__":
    all_superpixel_areas = process_all_superpixels(SEGMENTS_DIR)
    compute_statistics(all_superpixel_areas)

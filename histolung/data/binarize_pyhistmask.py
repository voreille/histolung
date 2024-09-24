from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm

# Use the project directory where the datasets are stored
project_dir = Path(__file__).parents[2]


def check_corners(img):
    """
    This function checks the corner pixels of an image and returns the pixel value (BGR) of the background.

    Parameters
    ----------
    img (numpy.ndarray): Image data

    Returns
    -------
    background_pixel (numpy.ndarray): pixel value (BGR) for the background
    """
    copy = img.copy()
    width, height, _ = copy.shape
    if width > 15000 or height > 15000:
        cropped_image = copy[600:width - 600, 600:height - 600]
    else:
        cropped_image = copy[300:width - 300, 300:height - 300]
    width, height, _ = cropped_image.shape
    top_left = img[0, 0, :]
    top_right = img[width - 1, 0, :]
    bottom_left = img[0, height - 1, :]
    bottom_right = img[width - 1, height - 1, :]
    most_frequent = np.argmax(
        np.bincount([
            np.sum(top_left),
            np.sum(top_right),
            np.sum(bottom_left),
            np.sum(bottom_right)
        ]))

    if most_frequent == np.sum(top_left):
        return top_left

    elif most_frequent == np.sum(top_right):
        return top_right

    elif most_frequent == np.sum(bottom_left):
        return bottom_left

    elif most_frequent == np.sum(bottom_right):
        return bottom_right


def binarze_mask_pyhist():
    """
    Convert PyHIST generated masks to binary masks and save them in the same directory.
    """
    # Define the mask directory
    maskdir = project_dir / "data/interim"

    # Loop over datasets (dataset_name)
    datasets = [d for d in maskdir.iterdir() if d.is_dir()]

    for dataset in datasets:
        # Loop over each image directory (image_name)
        image_dirs = [
            img_dir for img_dir in dataset.iterdir() if img_dir.is_dir()
        ]

        for img_dir in tqdm(image_dirs,
                            desc=f"Processing dataset: {dataset.stem}"):
            ppm_file = img_dir / f"segmented_{img_dir.stem}.ppm"

            if ppm_file.exists():
                # Read the segmented .ppm mask
                color_mask = cv.imread(str(ppm_file))

                # Mask from PyHIST to binary mask
                copy = color_mask.copy()

                # Use your utility function to check for corners
                most_frequent = check_corners(copy)
                copy[(copy != most_frequent).any(axis=-1)] = 1
                copy[(copy == most_frequent).all(axis=-1)] = 0
                binary_mask = copy[:, :, 0]

                # Define output binary mask path
                binary_mask_path = img_dir / f"binary_{img_dir.stem}.png"

                # Save the binary mask
                cv.imwrite(str(binary_mask_path), binary_mask,
                           [cv.IMWRITE_PNG_BILEVEL, 1])

    print(
        f"Binary masks created and saved in the respective directories under {maskdir}"
    )


def main():
    binarze_mask_pyhist()


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd
import numpy as np
import cv2 as cv


def compute_xy_coordinates(df, downsample_factor=1):
    df = df.copy()
    widths = df[df["Row"] == 0]["Width"].values * downsample_factor
    heights = df[df["Column"] == 0]["Height"].values * downsample_factor

    df["coord_x"] = df["Column"].apply(lambda index: np.sum(widths[:index]))
    df["coord_y"] = df["Row"].apply(lambda index: np.sum(heights[:index]))

    return df


def binarize_mask(ppm_file, output_dir=None):
    """
    Convert PyHIST generated masks to binary masks and save them  the same directory.
    """

    ppm_file = Path(ppm_file).resolve()

    if output_dir is None:
        output_dir = ppm_file.parent
    else:
        output_dir = Path(output_dir).resolve()

    # Read the segmented .ppm mask
    color_mask = cv.imread(str(ppm_file))

    # Mask from PyHIST to binary mask
    copy = color_mask.copy()

    # Use your utility function to check for corners
    most_frequent = check_corners(copy)
    copy[(copy != most_frequent).any(axis=-1)] = 1
    copy[(copy == most_frequent).all(axis=-1)] = 0
    binary_mask = copy[:, :, 0]

    # Save the binary mask
    filename = ppm_file.name.split(".")[0]
    output_path = Path(output_dir) / f"{filename}.png"
    cv.imwrite(str(output_path), binary_mask, [cv.IMWRITE_PNG_BILEVEL, 1])
    return output_path


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

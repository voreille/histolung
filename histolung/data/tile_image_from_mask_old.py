import logging
from pathlib import Path
from itertools import product

import numpy as np
from openslide import OpenSlide
from PIL import Image
from tqdm import tqdm


def main(
    wsi_path,
    mask_path,
    output_dir=None,
    magnification=10,
    tile_size=224,
    threshold=0.8,
    save_mask=False,
):
    wsi = OpenSlide(str(wsi_path))
    mask = Image.open(str(mask_path))
    if save_mask:
        output_dir_mask = Path(output_dir) / f"{Path(wsi_path).stem}_mask"
        output_dir_mask.mkdir(exist_ok=True)

    output_dir = Path(output_dir) / f"{Path(wsi_path).stem}_sp"
    output_dir.mkdir(exist_ok=True)

    base_magnification = get_base_magnification(wsi=wsi)

    # Calculate the downsampling factor
    downsample_factor = base_magnification / magnification

    # Find the closest level in OpenSlide for the desired magnification
    level, level_tile_size = get_best_level(wsi, tile_size, downsample_factor)
    level0_tile_size = int(np.ceil(tile_size * downsample_factor))

    mask_size = np.array(mask.size)
    mask_scale_factor = np.mean(np.array(wsi.dimensions) / mask_size)
    mask_array = np.array(mask)
    mask_array = (mask_array != 0)

    # Get WSI dimensions at level 0
    wsi_width, wsi_height = wsi.dimensions

    mask_tile_size = int(level0_tile_size // mask_scale_factor)
    coordinates = list(
        product(
            range(0, wsi_height, level0_tile_size),
            range(0, wsi_width, level0_tile_size),
        ))

    # Iterate through WSI in steps of tile_size
    for y, x in tqdm(coordinates,
                     desc=f"Processing tiles for image {Path(wsi_path).stem}"):
        # Extract the corresponding region in the mask
        mask_x = int(x // mask_scale_factor)
        mask_y = int(y // mask_scale_factor)

        # Get the mask patch (clipping to avoid going out of bounds)
        mask_patch = mask_array[mask_y:mask_y + mask_tile_size,
                                mask_x:mask_x + mask_tile_size]

        # Calculate the coverage of the mask in this region
        coverage = np.mean(mask_patch > 0)

        # If coverage meets the threshold, save the WSI tile
        if coverage >= threshold:
            tile = wsi.read_region(
                (x, y),
                level,
                (level_tile_size, level_tile_size),
            )
            if level_tile_size != tile_size:
                tile.resize((tile_size, tile_size), Image.LANCZOS)
            tile = tile.convert('RGB')  # Convert to RGB if necessary

            # Save the tile as a PNG
            tile_filename = f'tile_x{x}_y{y}.png'
            tile_path = Path(output_dir) / tile_filename
            tile.save(tile_path)

            if save_mask:
                # Save the mask patch as a separate PNG
                mask_patch_image = Image.fromarray(
                    (mask_patch * 255).astype(np.uint8))
                mask_tile_filename = f'mask_x{x}_y{y}.png'
                mask_tile_path = Path(output_dir_mask) / mask_tile_filename
                mask_patch_image = mask_patch_image.resize(
                    (tile_size, tile_size), Image.NEAREST)
                mask_patch_image.save(mask_tile_path)

    wsi.close()


def get_base_magnification(wsi: OpenSlide):
    # Common metadata keys to check for base magnification
    magnification_keys = [
        "aperio.AppMag",
        "openslide.objective-power",
        "hamamatsu.XResolution",
        "hamamatsu.ObjectiveMagnification",
    ]

    # Retrieve the base magnification
    base_magnification = None
    for key in magnification_keys:
        mag = wsi.properties.get(key)
        if mag:
            base_magnification = float(mag)
            break

    # Default to a common magnification if none found
    if base_magnification is None:
        base_magnification = 40.0
        logging.warning(f"Base magnifincation not found for {wsi_path.name}")

    return base_magnification


def get_level_for_magnification(
    wsi: OpenSlide,
    desired_magnification: float,
) -> int:
    # Common metadata keys to check for base magnification
    base_magnification = get_base_magnification(wsi=wsi)

    # Default to a common magnification if none found
    if base_magnification is None:
        base_magnification = 40.0

    # Calculate the downsampling factor
    downsample_factor = base_magnification / desired_magnification

    # Find the closest level in OpenSlide for the desired magnification
    level = wsi.get_best_level_for_downsample(downsample_factor)

    return level, downsample_factor


def get_best_level(wsi, tile_size, downsample_factor):
    level = wsi.get_best_level_for_downsample(downsample_factor)
    downsample_level = wsi.level_downsamples[level]
    level_tile_size = int(
        np.ceil(tile_size * downsample_factor / downsample_level))

    if level_tile_size != tile_size:
        downsample_level = wsi.level_downsamples[level + 1]
        level_p1_tile_size = int(
            np.ceil(tile_size * downsample_factor / downsample_level))
        if level_p1_tile_size == tile_size:
            level += 1
            level_tile_size = level_p1_tile_size
    return level, level_tile_size


if __name__ == "__main__":
    mask_path = "/home/valentin/workspaces/histolung/data/interim/masks/tcga_lusc/output/TCGA-37-A5EL-01A-01-TSA.58C8AF98-6A87-4C7A-AFB6-4CB0F9511F9B.svs/TCGA-37-A5EL-01A-01-TSA.58C8AF98-6A87-4C7A-AFB6-4CB0F9511F9B.svs_mask_use.png"
    wsi_path = "/home/valentin/workspaces/histolung/data/debug_histoqc/LUSC/ffd7fa3d-f0dc-4d23-b989-52abc8cbbb13/TCGA-37-A5EL-01A-01-TSA.58C8AF98-6A87-4C7A-AFB6-4CB0F9511F9B.svs"
    output_dir = "/home/valentin/workspaces/histolung/data/test_histoqc/test_tiling"
    main(wsi_path, mask_path, output_dir=output_dir)

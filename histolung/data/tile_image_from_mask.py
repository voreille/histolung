import logging
from pathlib import Path
from itertools import product
from multiprocessing.pool import ThreadPool

import numpy as np
from openslide import OpenSlide
from PIL import Image
from tqdm import tqdm

DEBUG = False


class TileProcessor:

    def __init__(self, wsi_path, mask_path, output_dir, magnification,
                 tile_size, threshold, save_mask):
        # Initialize with relevant parameters
        self.wsi = OpenSlide(wsi_path)  # OpenSlide object for tile extraction
        self.mask_array = np.array(Image.open(mask_path)) != 0
        self.output_dir = Path(output_dir) / Path(wsi_path).stem
        self.output_dir_mask = self.output_dir / "mask" if save_mask else None
        self.magnification = magnification
        self.tile_size = tile_size
        self.threshold = threshold
        self.save_mask = save_mask

        # Initialize directories
        self.output_dir.mkdir(exist_ok=True)
        if save_mask:
            self.output_dir_mask.mkdir(exist_ok=True)

        # Compute level and scaling factors
        self.base_magnification = self.get_base_magnification()
        downsample_factor = self.base_magnification / magnification
        self.level, self.level_tile_size = self.get_best_level(
            downsample_factor)
        self.level0_tile_size = int(np.ceil(tile_size * downsample_factor))

        # Calculate scale factors for the mask alignment
        self.mask_scale_factor = self._get_mask_scale_factor()

        self.mask_tile_size = int(self.level0_tile_size //
                                  self.mask_scale_factor)

    def _get_mask_scale_factor(self):
        # Compute level and scaling factors
        self.base_magnification = self.get_base_magnification()
        downsample_factor = self.base_magnification / self.magnification
        self.level, self.level_tile_size = self.get_best_level(
            downsample_factor)

        # Calculate separate scale factors for width and height
        wsi_width, wsi_height = self.wsi.dimensions
        mask_width, mask_height = self.mask_array.T.shape

        scale_factor_width = wsi_width / mask_width
        scale_factor_height = wsi_height / mask_height

        # Check if the scale factors are close to each other
        if not np.isclose(scale_factor_width, scale_factor_height, rtol=1e-3):
            raise ValueError(
                f"Scale factors for width ({scale_factor_width}) and height ({scale_factor_height}) differ significantly."
            )

        # Use the mean scale factor if they are close
        return np.mean([scale_factor_width, scale_factor_height])

    def get_base_magnification(self):
        """Retrieve base magnification from WSI metadata."""
        magnification_keys = [
            "aperio.AppMag",
            "openslide.objective-power",
            "hamamatsu.XResolution",
            "hamamatsu.ObjectiveMagnification",
        ]
        for key in magnification_keys:
            mag = self.wsi.properties.get(key)
            if mag:
                return float(mag)
        logging.warning("Base magnification not found; defaulting to 40x.")
        return 40.0

    def get_best_level(self, downsample_factor):
        """Find the best level in OpenSlide for the desired magnification."""
        level = self.wsi.get_best_level_for_downsample(downsample_factor)
        downsample_level = self.wsi.level_downsamples[level]
        level_tile_size = int(
            np.ceil(self.tile_size * downsample_factor / downsample_level))

        if level_tile_size != self.tile_size:
            downsample_level = self.wsi.level_downsamples[level + 1]
            level_p1_tile_size = int(
                np.ceil(self.tile_size * downsample_factor / downsample_level))
            if level_p1_tile_size == self.tile_size:
                level += 1
                level_tile_size = level_p1_tile_size
        return level, level_tile_size

    def get_coordinates(self):
        return list(
            product(
                range(0, self.wsi.dimensions[1], self.level0_tile_size),
                range(0, self.wsi.dimensions[0], self.level0_tile_size),
            ))

    def __call__(self, coords):
        y, x = coords  # Coordinates for the tile

        # Calculate the corresponding region in the mask
        mask_x = int(x // self.mask_scale_factor)
        mask_y = int(y // self.mask_scale_factor)

        # Get the mask patch (clipping to avoid out of bounds)
        mask_patch = self.mask_array[mask_y:mask_y + self.mask_tile_size,
                                     mask_x:mask_x + self.mask_tile_size]
        coverage = np.mean(mask_patch > 0)

        # If coverage meets the threshold, save the WSI tile
        if coverage >= self.threshold:
            tile = self.wsi.read_region(
                (x, y),
                self.level,
                (self.level_tile_size, self.level_tile_size),
            )
            if self.level_tile_size != self.tile_size:
                tile = tile.resize((self.tile_size, self.tile_size),
                                   Image.LANCZOS)
            tile = tile.convert('RGB')  # Convert to RGB if necessary

            # Save the tile as a PNG
            tile_filename = f'tile_x{x}_y{y}.png'
            tile_path = self.output_dir / tile_filename
            tile.save(tile_path)

            # Optionally save mask tile
            if self.save_mask:
                mask_patch_image = Image.fromarray(
                    (mask_patch * 255).astype(np.uint8))
                mask_patch_image = mask_patch_image.resize(
                    (self.tile_size, self.tile_size), Image.NEAREST)
                mask_tile_filename = f'mask_x{x}_y{y}.png'
                mask_tile_path = self.output_dir_mask / mask_tile_filename
                mask_patch_image.save(mask_tile_path)


def main(
    wsi_path,
    mask_path,
    output_dir=None,
    magnification=40,
    tile_size=224,
    threshold=0.8,
    num_workers=12,
    save_mask=False,
):
    tile_processor = TileProcessor(wsi_path=wsi_path,
                                   mask_path=mask_path,
                                   output_dir=output_dir,
                                   magnification=magnification,
                                   tile_size=tile_size,
                                   threshold=threshold,
                                   save_mask=save_mask)

    # Define coordinates for tiles
    coordinates = tile_processor.get_coordinates()

    if DEBUG:
        for coord in tqdm(coordinates):
            tile_processor(coord)
        return

    # Process tiles in parallel using ThreadPool
    with ThreadPool(processes=num_workers) as pool:
        list(
            tqdm(pool.imap_unordered(tile_processor, coordinates),
                 total=len(coordinates),
                 desc="Processing tiles"))


if __name__ == "__main__":
    mask_path = "/home/valentin/workspaces/histolung/data/interim/masks/tcga_lusc/output/TCGA-37-A5EL-01A-01-TSA.58C8AF98-6A87-4C7A-AFB6-4CB0F9511F9B.svs/TCGA-37-A5EL-01A-01-TSA.58C8AF98-6A87-4C7A-AFB6-4CB0F9511F9B.svs_mask_use.png"
    wsi_path = "/home/valentin/workspaces/histolung/data/debug_histoqc/LUSC/ffd7fa3d-f0dc-4d23-b989-52abc8cbbb13/TCGA-37-A5EL-01A-01-TSA.58C8AF98-6A87-4C7A-AFB6-4CB0F9511F9B.svs"
    output_dir = "/home/valentin/workspaces/histolung/data/test_histoqc/test_tiling"
    main(wsi_path, mask_path, output_dir=output_dir)

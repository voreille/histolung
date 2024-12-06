import logging
from pathlib import Path
from itertools import product
from multiprocessing.pool import ThreadPool
import json

import numpy as np
import pandas as pd
from openslide import OpenSlide
from PIL import Image
from tqdm import tqdm

DEBUG = False


class WSITilerWithMask:

    def __init__(self,
                 wsi_path,
                 mask_path,
                 output_dir,
                 magnification=10,
                 tile_size=224,
                 threshold=0.8,
                 save_mask=False,
                 raise_error_mag=True,
                 save_tile_overlay=False):
        # Initialize parameters and objects
        self.wsi = OpenSlide(wsi_path)
        self.mask_array = np.array(Image.open(mask_path)) != 0
        self.wsi_id = Path(mask_path).stem.split(".")[0]
        self.output_dir = Path(output_dir) / self.wsi_id
        self.tiles_output_dir = self.output_dir / "tiles"

        self.output_dir_mask = self.output_dir / "mask" if save_mask else None
        self.magnification = magnification
        self.tile_size = tile_size
        self.threshold = threshold
        self.save_mask = save_mask
        self.save_tile_overlay = save_tile_overlay
        self.raise_error_mag = raise_error_mag
        self.patch_metadata = []
        self.x_px_size_level0 = float(
            self.wsi.properties.get("openslide.mpp-x", "nan"))
        self.y_px_size_level0 = float(
            self.wsi.properties.get("openslide.mpp-y", "nan"))

        # Directories setup
        self.output_dir.mkdir(exist_ok=True)
        self.tiles_output_dir.mkdir(exist_ok=True)
        if save_mask:
            self.output_dir_mask.mkdir(exist_ok=True)

        # Scaling calculations
        self.base_magnification = self.get_base_magnification()
        self.downsample_factor = self.base_magnification / magnification
        self.level, self.level_tile_size = self.get_best_level()

        self.level0_tile_size = int(np.ceil(tile_size *
                                            self.downsample_factor))

        # Mask alignment scale factor
        self.mask_scale_factor = self._get_mask_scale_factor()
        self.mask_tile_size = int(
            round(self.level0_tile_size / self.mask_scale_factor))

        # Create the overlay for visualizing selected tiles on WSI with mask if enabled
        if save_tile_overlay:
            mask_height, mask_width = self.mask_array.shape

            # Create WSI thumbnail at the mask scale
            self.wsi_thumbnail = self.wsi.get_thumbnail(
                (mask_width, mask_height)).convert("RGB")

            # mask_width, mask_height = self.wsi_thumbnail.size

            # # Resize mask to match thumbnail size and create an RGB mask
            # mask_resized = Image.fromarray(
            #     (self.mask_array * 255).astype(np.uint8)).resize(
            #         (mask_width, mask_height), Image.NEAREST)

            # # Convert mask to binary (0 or 1) for multiplication
            # mask_binary = np.array(mask_resized) / 255
            # mask_rgb = np.stack([mask_binary] * 3, axis=-1)

            # # Apply the mask by multiplying with the thumbnail

            # thumbnail_array = np.array(self.wsi_thumbnail)
            # masked_thumbnail = (thumbnail_array * mask_rgb).astype(np.uint8)
            # self.wsi_thumbnail = Image.fromarray(masked_thumbnail)

    def _get_mask_scale_factor(self):
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
        """Retrieve base magnification from WSI metadata or infer it from MPP."""
        # Check if magnification is available
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

        if self.raise_error_mag:
            raise ValueError(
                f"Magnification metadata is missing for WSI {self.wsi_path.name}. "
                "Please ensure the WSI has magnification information or "
                "set `raise_error_mag` to False to attempt inference.")

        # Attempt to infer magnification based on MPP if not available
        mpp_x = float(self.wsi.properties.get("openslide.mpp-x", "nan"))
        reference_mpp_40x = 0.25  # Assume 0.25 microns/pixel for 40x as a reference

        if not np.isnan(mpp_x):
            estimated_magnification = reference_mpp_40x / mpp_x * 40
            logging.warning(
                f"Inferred magnification from MPP as {estimated_magnification:.2f}x based on MPP: {mpp_x}."
            )
            return estimated_magnification
        else:
            logging.warning("Base magnification not found; defaulting to 40x.")
            return 40.0

    def get_best_level(self):
        """Find the best level in OpenSlide for the desired magnification."""
        level = self.wsi.get_best_level_for_downsample(self.downsample_factor)
        downsample_level = self.wsi.level_downsamples[level]
        level_tile_size = int(
            np.ceil(self.tile_size * self.downsample_factor /
                    downsample_level))

        if level_tile_size != self.tile_size:
            downsample_level = self.wsi.level_downsamples[level + 1]
            level_p1_tile_size = int(
                np.ceil(self.tile_size * self.downsample_factor /
                        downsample_level))
            if level_p1_tile_size == self.tile_size:
                level += 1
                level_tile_size = level_p1_tile_size
        return level, level_tile_size

    def get_coordinates(self):
        """
        Return the coordinate of each potential complete 
        squared tile at level 0
        """
        return list(
            product(
                range(
                    0,
                    self.wsi.dimensions[1] - self.level0_tile_size + 1,
                    self.level0_tile_size,
                ),
                range(
                    0,
                    self.wsi.dimensions[0] - self.level0_tile_size + 1,
                    self.level0_tile_size,
                ),
            ))

    def __call__(self, coords):
        y, x = coords  # Coordinates for the tile

        tile_id = f'{self.wsi_id}__x{x}_y{y}'
        # Calculate the corresponding region in the mask
        mask_x = int(x // self.mask_scale_factor)
        mask_y = int(y // self.mask_scale_factor)

        # Get the mask patch (clipping to avoid out of bounds)
        mask_patch = self.mask_array[mask_y:mask_y + self.mask_tile_size,
                                     mask_x:mask_x + self.mask_tile_size]
        coverage = np.mean(mask_patch > 0)

        # If coverage meets the threshold, save the WSI tile
        keep = 0
        if coverage >= self.threshold:

            keep = 1
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
            tile_path = self.tiles_output_dir / f"{tile_id}.png"
            tile.save(tile_path)

            # Outline the selected tiles on the thumbnail overlay
            if self.save_tile_overlay:
                outline = Image.new("RGBA",
                                    (self.mask_tile_size, self.mask_tile_size),
                                    (0, 255, 0, 128))
                self.wsi_thumbnail.paste(outline, (mask_x, mask_y), outline)

            # Optionally save mask tile
            if self.save_mask:
                mask_patch_image = Image.fromarray(
                    (mask_patch * 255).astype(np.uint8))
                mask_patch_image = mask_patch_image.resize(
                    (self.tile_size, self.tile_size), Image.NEAREST)
                mask_tile_filename = f'{self.wsi_id}__{x}_y{y}.png'
                mask_tile_path = self.output_dir_mask / mask_tile_filename
                mask_patch_image.save(mask_tile_path)

        # Save metadata
        self.patch_metadata.append({
            "tile_id":
            tile_id,
            "x_level0":
            x,
            "y_level0":
            y,
            "x_current_level":
            int(x // self.downsample_factor),
            "y_current_level":
            int(y // self.downsample_factor),
            "row":
            int(y // self.level0_tile_size),
            "column":
            int(x // self.level0_tile_size),
            "keep":
            keep,
            "mask_coverage_ratio":
            coverage,
        })

    def save_overlay(self):
        if self.save_tile_overlay:
            overlay_path = self.output_dir / f"{self.wsi_id}__tile_overlay.png"
            self.wsi_thumbnail.save(overlay_path)

    def save_metadata(self):
        """
        Save metadata to a CSV file using pandas.
        """
        # Convert the metadata list to a pandas DataFrame
        patch_metadata_df = pd.DataFrame(self.patch_metadata).sort_values(
            by=['row', 'column'], ascending=[True, True])

        # Save the DataFrame to a CSV file
        patch_metadata_df.to_csv(
            self.output_dir / f"{self.wsi_id}__tiling_results.csv",
            index=False,
        )

        metadata = {
            "tile_magnification": self.magnification,
            "base_magnification": self.base_magnification,
            "x_px_size_tile": self.x_px_size_level0 * self.downsample_factor,
            "y_px_size_tile": self.y_px_size_level0 * self.downsample_factor,
            "x_px_size_base": self.x_px_size_level0,
            "y_px_size_base": self.y_px_size_level0,
        }
        with open(self.output_dir / f"{self.wsi_id}__metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        logging.info(f"Metadata saved to {self.output_dir}")


def main(
    wsi_path,
    mask_path,
    output_dir=None,
    magnification=20,
    tile_size=224,
    threshold=0.5,
    num_workers=12,
    save_mask=False,
    save_tile_overlay=False,
):
    tile_processor = WSITilerWithMask(
        wsi_path=wsi_path,
        mask_path=mask_path,
        output_dir=output_dir,
        magnification=magnification,
        tile_size=tile_size,
        threshold=threshold,
        save_mask=save_mask,
        save_tile_overlay=save_tile_overlay,
    )

    # Define coordinates for tiles
    coordinates = tile_processor.get_coordinates()

    if DEBUG:
        for coord in tqdm(coordinates):
            tile_processor(coord)
        tile_processor.save_overlay()
        return

    # Process tiles in parallel using ThreadPool
    with ThreadPool(processes=num_workers) as pool:
        list(
            tqdm(pool.imap_unordered(tile_processor, coordinates),
                 total=len(coordinates),
                 desc="Processing tiles"))

    # Save the tile overlay after processing if enabled
    tile_processor.save_overlay()
    tile_processor.save_metadata(mask_path / "metadata.csv")


if __name__ == "__main__":
    mask_path = "/home/valentin/workspaces/histolung/data/interim/debug_masks/tcga_lusc/output/TCGA-37-A5EL-01A-01-TSA.58C8AF98-6A87-4C7A-AFB6-4CB0F9511F9B.svs/TCGA-37-A5EL-01A-01-TSA.58C8AF98-6A87-4C7A-AFB6-4CB0F9511F9B.svs_mask_use.png"
    wsi_path = "/home/valentin/workspaces/histolung/data/debug_histoqc/LUSC/ffd7fa3d-f0dc-4d23-b989-52abc8cbbb13/TCGA-37-A5EL-01A-01-TSA.58C8AF98-6A87-4C7A-AFB6-4CB0F9511F9B.svs"
    output_dir = "/home/valentin/workspaces/histolung/data/test_histoqc/test_tiling"
    main(wsi_path, mask_path, output_dir=output_dir, save_tile_overlay=True)

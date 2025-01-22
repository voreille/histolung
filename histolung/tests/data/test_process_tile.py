# TODO: test if the tiles are adjacent
# TODO: test other stuff I think...

import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch
from histolung.data.wsi_tiler import WSITilerWithMask


class TestTileProcessor(unittest.TestCase):

    def setUp(self):
        # Define dummy paths for WSI, mask, and output directory
        self.wsi_path = Path("/mock/path/to/test_WSI.svs")
        self.mask_path = Path("/mock/path/to/mask_use.png")
        self.output_dir = Path("/mock/path/to/output")
        self.magnification = 10
        self.tile_size = 224
        self.threshold = 0.8
        self.save_mask = False
        self.save_tile_overlay = False
        self.wsi_width, self.wsi_height = (
            33152, 81856)  # Calculated to be 32x mask dimensions
        self.mask_width, self.mask_height = (1036, 2558)
        self.mask_scale_factor = 32  # 32x scaling factor

        # Mock OpenSlide and Path methods
        with patch("histolung.data.tile_image_from_mask.OpenSlide") as MockOpenSlide, \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"), \
             patch("PIL.Image.open"), \
             patch("numpy.array", return_value=np.ones((self.mask_height, self.mask_width), dtype=bool)), \
             patch.object(WSITilerWithMask, 'get_base_magnification', return_value=40.0):

            # Configure MockOpenSlide with necessary properties
            MockOpenSlide.return_value.dimensions = (self.wsi_width,
                                                     self.wsi_height)
            MockOpenSlide.return_value.get_best_level_for_downsample.return_value = 0
            MockOpenSlide.return_value.level_downsamples = (
                1.0,
                4.0000603209072265,
                16.001206563706564,
                32.005540570728215,
            )

            # Initialize TileProcessor with the mocked paths and parameters
            self.tile_processor = WSITilerWithMask(
                wsi_path=self.wsi_path,
                mask_path=self.mask_path,
                output_dir=self.output_dir,
                magnification=self.magnification,
                tile_size=self.tile_size,
                threshold=self.threshold,
                save_masks=self.save_mask,
                save_tile_overlay=self.save_tile_overlay)

    def test_initialization(self):
        """Test that TileProcessor is initialized correctly with given parameters."""
        self.assertEqual(self.tile_processor.magnification, self.magnification)
        self.assertEqual(self.tile_processor.tile_size, self.tile_size)
        self.assertEqual(self.tile_processor.threshold, self.threshold)
        self.assertIsInstance(self.tile_processor.output_dir, Path)

    def test_base_magnification(self):
        """Test that the base magnification is retrieved or defaults to 40x if not found."""
        self.assertEqual(self.tile_processor.base_magnification, 40.0)

    def test_mask_scale_factor(self):
        """Test that the mask scale factor is calculated correctly."""
        # Assume WSI and mask are the same size for simplicity in this test
        scale_factor = self.tile_processor._get_mask_scale_factor()
        self.assertAlmostEqual(scale_factor, self.mask_scale_factor, places=3)

    def test_tile_coordinates_generation(self):
        """Test that tile coordinates are generated correctly to cover the WSI without gaps."""
        coordinates = self.tile_processor.get_coordinates()
        expected_rows = int(
            np.floor(self.wsi_height / self.tile_processor.level0_tile_size))
        expected_cols = int(
            np.floor(self.wsi_width / self.tile_processor.level0_tile_size))
        self.assertEqual(len(coordinates), expected_rows * expected_cols)

    def test_mask_tile_size_calculation(self):
        """Test that the mask_tile_size is calculated based on scale factor and tile size."""
        expected_mask_tile_size = round(self.tile_processor.level0_tile_size //
                                        self.tile_processor.mask_scale_factor)
        self.assertEqual(self.tile_processor.mask_tile_size,
                         expected_mask_tile_size)

    def test_get_best_level(self):
        """Test the get_best_level method returns a valid level and tile size."""
        level, level_tile_size = self.tile_processor.get_best_level(1.0)
        self.assertEqual(level, 0)
        self.assertEqual(level_tile_size, self.tile_processor.tile_size)


if __name__ == '__main__':
    unittest.main()

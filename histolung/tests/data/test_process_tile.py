import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
from PIL import Image
from ProcessTile import ProcessTile  # Assuming ProcessTile is in ProcessTile.py

class TestProcessTile(unittest.TestCase):
    
    def setUp(self):
        # Initialize parameters
        self.wsi_path = "test_wsi.svs"
        self.mask_path = "test_mask.png"
        self.output_dir = "output"
        self.magnification = 10
        self.tile_size = 224
        self.threshold = 0.8
        self.save_mask = True
        
        # Mock the OpenSlide object
        self.mock_wsi = MagicMock()
        self.mock_wsi.dimensions = (1000, 1000)
        self.mock_wsi.level_downsamples = [1, 2, 4]
        self.mock_wsi.get_best_level_for_downsample.return_value = 1
        self.mock_wsi.level_dimensions = [(1000, 1000), (500, 500), (250, 250)]
        
        # Patch OpenSlide and Image.open
        patcher_wsi = patch('openslide.OpenSlide', return_value=self.mock_wsi)
        patcher_image_open = patch('PIL.Image.open', return_value=Image.fromarray(np.zeros((100, 100), dtype=np.uint8)))
        self.mock_openslide = patcher_wsi.start()
        self.mock_image_open = patcher_image_open.start()
        
        self.addCleanup(patcher_wsi.stop)
        self.addCleanup(patcher_image_open.stop)

    def test_initialization(self):
        # Initialize the ProcessTile object and check attributes
        processor = ProcessTile(
            wsi_path=self.wsi_path,
            mask_path=self.mask_path,
            output_dir=self.output_dir,
            magnification=self.magnification,
            tile_size=self.tile_size,
            threshold=self.threshold,
            save_mask=self.save_mask
        )
        self.assertEqual(processor.base_magnification, 40.0)  # Default mock magnification
        self.assertEqual(processor.level, 1)
        self.assertEqual(processor.level_tile_size, processor.tile_size // 2)  # Assuming downsample of 2 for level 1
        self.assertTrue(processor.output_dir.exists())

    def test_tile_saving(self):
        # Mock the save method of PIL.Image
        with patch('PIL.Image.Image.save') as mock_save:
            processor = ProcessTile(
                wsi_path=self.wsi_path,
                mask_path=self.mask_path,
                output_dir=self.output_dir,
                magnification=self.magnification,
                tile_size=self.tile_size,
                threshold=self.threshold,
                save_mask=self.save_mask
            )

            # Mock the region reading and mask tile creation
            self.mock_wsi.read_region.return_value = Image.new("RGB", (processor.level_tile_size, processor.level_tile_size))
            
            # Call the tile processor with a sample coordinate
            processor((0, 0))
            
            # Assert save was called for the tile and optionally for the mask
            self.assertTrue(mock_save.called)
            if self.save_mask:
                self.assertEqual(mock_save.call_count, 2)  # Once for tile, once for mask
            else:
                self.assertEqual(mock_save.call_count, 1)  # Only for tile

    def test_mask_thresholding(self):
        # Test with mock mask having some coverage over threshold
        processor = ProcessTile(
            wsi_path=self.wsi_path,
            mask_path=self.mask_path,
            output_dir=self.output_dir,
            magnification=self.magnification,
            tile_size=self.tile_size,
            threshold=0.5,  # Lower threshold for test
            save_mask=self.save_mask
        )

        # Mock a mask array where only part of the area is non-zero (coverage > threshold)
        processor.mask_array = np.ones((100, 100), dtype=bool)
        processor.mask_array[:50, :50] = 0  # 75% coverage

        # Verify that the tile is saved as coverage > threshold
        with patch('PIL.Image.Image.save') as mock_save:
            processor((0, 0))
            self.assertTrue(mock_save.called)  # Tile should be saved since coverage > threshold

    def test_get_best_level(self):
        # Verify correct level selection logic based on downsampling factor
        processor = ProcessTile(
            wsi_path=self.wsi_path,
            mask_path=self.mask_path,
            output_dir=self.output_dir,
            magnification=self.magnification,
            tile_size=self.tile_size,
            threshold=self.threshold,
            save_mask=self.save_mask
        )

        # Mock specific downsampling factor for testing
        processor.base_magnification = 20.0
        downsample_factor = processor.base_magnification / processor.magnification

        level, tile_size = processor.get_best_level(downsample_factor)
        self.assertEqual(level, 1)  # Expect level 1 based on mock setup
        self.assertEqual(tile_size, processor.tile_size // 2)  # Check the computed tile size

if __name__ == "__main__":
    unittest.main()

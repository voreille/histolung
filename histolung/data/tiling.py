import logging
from pathlib import Path
from multiprocessing.pool import ThreadPool

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from histolung.data.wsi_tiler import WSITilerWithMask
from histolung.utils.yaml import load_yaml_with_env

load_dotenv()

DEBUG = False
project_dir = Path(__file__).parents[2].resolve()

config = load_yaml_with_env(project_dir /
                            "histolung/config/datasets_config.yaml")

output_basedir = project_dir / ("data/interim/debug_tiles"
                                if DEBUG else config["tiles_basedir"])
output_basedir.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_wsi(
    wsi_path,
    mask_path,
    output_dir=None,
    magnification=10,
    tile_size=224,
    threshold=0.8,
    save_mask=False,
    save_tile_overlay=False,
    num_workers=24,
    save_metadata=True,
):
    try:
        tile_processor = WSITilerWithMask(
            wsi_path,
            mask_path,
            output_dir,
            magnification=magnification,
            tile_size=tile_size,
            threshold=threshold,
            save_mask=save_mask,
            save_tile_overlay=save_tile_overlay,
        )

        # Define coordinates for tiles
        coordinates = tile_processor.get_coordinates()

        # Multiprocessing for tile extraction with error handling
        if num_workers>1:
            with ThreadPool(processes=num_workers) as pool:
                results = list(
                    tqdm(pool.imap_unordered(tile_processor, coordinates),
                        total=len(coordinates),
                        desc="Processing tiles"))
        else:
            for coor in coordinates:
                tile_processor(coor)
            

        # Check if any results returned an exception
        if any(isinstance(res, Exception) for res in results):
            raise RuntimeError(
                f"Error encountered in tile processing for {wsi_path.name}")

        logger.info(f"Tiling completed for WSI {wsi_path.name}")

    except Exception as e:
        logger.error(f"Error processing WSI {wsi_path.name}: {e}")
        return False  # Return False to indicate failure

    if save_tile_overlay:
        tile_processor.save_overlay()

    if save_metadata:
        tile_processor.save_metadata()

    return True  # Return True to indicate success


def tile_dataset(
    masks_dir,
    output_dir,
    tile_size=224,
    threshold=0.8,
    num_workers=1,
    save_tile_overlay=False,
    debug_id=None,
):
    mask_files = [f for f in masks_dir.rglob("*mask_use.png")]
    df = pd.read_csv(Path(masks_dir) / "raw_wsi_path.csv")
    wsi_paths_mapping = {
        row["WSI_ID"]: Path(row["Path"])
        for _, row in df.iterrows()
    }

    if debug_id:
        # to debug a specific wsi
        mask_path = [m for m in mask_files if debug_id in str(m)][0]
        wsi_path = wsi_paths_mapping[debug_id]
        success = process_wsi(
            wsi_path,
            mask_path,
            output_dir=output_dir,
            magnification=10,
            tile_size=tile_size,
            threshold=threshold,
            num_workers=num_workers,
            save_tile_overlay=save_tile_overlay,
        )
        return

    for mask_path in mask_files:
        wsi_id = mask_path.name.split(".")[0]
        wsi_path = wsi_paths_mapping[wsi_id]
        success = process_wsi(
            wsi_path,
            mask_path,
            output_dir=output_dir,
            magnification=10,
            tile_size=tile_size,
            threshold=threshold,
            num_workers=num_workers,
            save_tile_overlay=save_tile_overlay,
        )

        if not success:
            logger.warning(f"Processing failed for WSI {wsi_path.name}")

import logging
from pathlib import Path
from multiprocessing.pool import ThreadPool, Pool

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
    num_workers_tiles=12,
    save_mask=False,
    save_tile_overlay=False,
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
        if num_workers_tiles > 1:
            with ThreadPool(processes=num_workers_tiles) as pool:
                results = list(
                    tqdm(pool.imap_unordered(tile_processor, coordinates),
                         total=len(coordinates),
                         desc="Processing tiles"))
        else:
            results = []
            for coord in tqdm(coordinates):
                result = tile_processor(coord)
                results.append(result)

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
    num_workers_wsi=4,
    num_workers_tiles=12,
    save_tile_overlay=False,
    magnification=10,
    debug_id=None,
):
    mask_files = [f for f in masks_dir.rglob("*mask_use.png")]
    df = pd.read_csv(Path(masks_dir) / "raw_wsi_path.csv")
    wsi_paths_mapping = {
        row["WSI_ID"]: Path(row["Path"])
        for _, row in df.iterrows()
    }

    if debug_id:
        # Debug a specific WSI
        mask_path = [m for m in mask_files if debug_id in str(m)][0]
        wsi_path = wsi_paths_mapping[debug_id]
        process_wsi(
            wsi_path,
            mask_path,
            output_dir=output_dir,
            magnification=magnification,
            tile_size=tile_size,
            threshold=threshold,
            num_workers_tiles=num_workers_tiles,
            save_tile_overlay=save_tile_overlay,
        )
        return

    # Create a list of arguments for parallel processing
    wsi_args = [
        (
            wsi_paths_mapping[mask_path.stem.split(".")[0]],  # WSI path
            mask_path,  # Mask path
            output_dir,
            magnification,
            tile_size,
            threshold,
            num_workers_tiles,
            False,  # save_mask
            save_tile_overlay,
        ) for mask_path in mask_files
    ]

    # Process WSIs in parallel
    with Pool(processes=num_workers_wsi) as pool:
        results = list(
            tqdm(pool.starmap(process_wsi, wsi_args),
                 total=len(wsi_args),
                 desc="Processing WSIs"))

    for wsi_path, result in zip(wsi_paths_mapping.values(), results):
        if not result:
            logger.warning(f"Processing failed for WSI {wsi_path.name}")


if __name__ == "__main__":
    masks_dir = Path(
        "/home/valentin/workspaces/histolung/data/interim/debug_masks/tcga_lusc/output"
    )
    output_dir = Path(
        "/home/valentin/workspaces/histolung/data/test_histoqc/test_tiling")
    tile_dataset(
        masks_dir,
        output_dir=output_dir,
        tile_size=224,
        threshold=0.8,
        num_workers_wsi=4,
        num_workers_tiles=12,
        magnification=20,
        save_tile_overlay=True,
    )

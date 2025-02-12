import os
import logging
from pathlib import Path
from multiprocessing.pool import ThreadPool, Pool
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from histolung.data.wsi_tiler import WSITilerWithSuperPixelMaskWithOverlap
from histolung.utils.yaml import load_yaml_with_env

load_dotenv()

DEBUG = False
project_dir = Path(__file__).parents[2].resolve()

config = load_yaml_with_env(project_dir /
                            "histolung/config/datasets_config.yaml")

output_basedir = project_dir / ("data/interim/debug_tiles"
                                if DEBUG else config["tiles_basedir"])
output_basedir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler("superpixel_tiling.log", mode="w")  # Logs to file
    ],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_wsi(
    wsi_path,
    mask_path,
    output_dir=None,
    magnification=10,
    tile_size=224,
    threshold=0.8,
    num_workers_tiles=12,
    save_tile_overlay=False,
    save_metadata=True,
    average_superpixel_area=1000,
    average_n_tiles=10,
):
    logger.info(f"Starting tiling for WSI {wsi_path.name}")
    try:
        tile_processor = WSITilerWithSuperPixelMaskWithOverlap(
            wsi_path,
            mask_path,
            output_dir,
            magnification=magnification,
            tile_size=tile_size,
            threshold=threshold,
            save_tile_overlay=save_tile_overlay,
            average_superpixel_area=average_superpixel_area,
            average_n_tiles=average_n_tiles,
        )
        superpixel_labels = tile_processor.labels

        if num_workers_tiles > 1:

            with ThreadPool(processes=num_workers_tiles) as pool:
                # results = list(
                #     tqdm(pool.imap_unordered(tile_processor, superpixel_labels),
                #         total=len(superpixel_labels),
                #         desc="Processing superpixels"))

                results = list(
                    pool.imap_unordered(
                        tile_processor,
                        superpixel_labels,
                    ))

        else:
            results = [
                tile_processor(label) for label in tqdm(superpixel_labels)
            ]

        if any(isinstance(res, Exception) for res in results):
            raise RuntimeError(
                f"Error encountered in tile processing for {wsi_path.name}")

        logger.info(f"Tiling completed for WSI {wsi_path.name}")

        if save_tile_overlay:
            tile_processor.save_overlay()
        if save_metadata:
            tile_processor.save_metadata()

    except Exception as e:
        logger.error(f"Error processing WSI {wsi_path.name}: {e}")
        return False

    return True


def process_wsi_wrapper(args):
    """Unpacks tuple args before calling process_wsi"""
    return process_wsi(*args)


def get_wsi_path_from_mask_path(mask_path, raw_data_dir):
    wsi_id = mask_path.stem.replace("_segments", "")
    match = list(raw_data_dir.rglob(f"{wsi_id}*.svs"))
    if len(match) > 1:
        raise ValueError(f"multiple matching *.svs for the wsi_id: {wsi_id}")

    if len(match) == 0:
        raise ValueError(f"No matching *.svs for wsi_id: {wsi_id}")

    return match[0]


def tile_dataset(
    raw_data_dir,
    masks_dir,
    output_dir,
    tile_size=224,
    threshold=0.8,
    num_workers_wsi=4,
    num_workers_tiles=12,
    save_tile_overlay=False,
    magnification=10,
    debug_id=None,
    average_superpixel_area=1000,
    average_n_tiles=10,
):

    mask_files = [f for f in masks_dir.rglob("*.tiff")]

    if debug_id:
        # Debug a specific WSI
        try:
            mask_path = [m for m in mask_files if debug_id in str(m)][0]
        except:
            raise ValueError(f"{debug_id} not found in this dataset!!")
        wsi_path = get_wsi_path_from_mask_path(mask_path, raw_data_dir)
        process_wsi(
            wsi_path,
            mask_path,
            output_dir=output_dir,
            magnification=magnification,
            tile_size=tile_size,
            threshold=threshold,
            num_workers_tiles=num_workers_tiles,
            save_tile_overlay=save_tile_overlay,
            average_superpixel_area=average_superpixel_area,
            average_n_tiles=average_n_tiles,
        )
        return

    # Create a list of arguments for parallel processing
    wsi_args = [(
        get_wsi_path_from_mask_path(mask_path, raw_data_dir),
        mask_path,
        output_dir,
        magnification,
        tile_size,
        threshold,
        num_workers_tiles,
        save_tile_overlay,
    ) for mask_path in mask_files]

    # with Pool(processes=num_workers_wsi) as pool:
    #     results = list(
    #         tqdm(pool.starmap(process_wsi, wsi_args),
    #              total=len(wsi_args),
    #              desc="Processing WSIs"))

    with Pool(processes=num_workers_wsi) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_wsi_wrapper, wsi_args),
                total=len(wsi_args),
                desc="Processing WSIs",
            ))

    for wsi_arg, result in zip(wsi_args, results):
        wsi_path = wsi_arg[0]
        if not result:
            logger.warning(f"Processing failed for WSI {wsi_path.name}")


if __name__ == "__main__":
    cptac_path = Path(os.getenv("CPTAC_DATA_RAW_PATH"))
    tcga_path = Path(os.getenv("TCGA_DATA_RAW_PATH"))
    project_dir = Path(__file__).parents[2].resolve()
    masks_path = project_dir / "data/interim/superpixels"
    output_basedir = project_dir / "data/interim/tiles_superpixels"
    output_basedir.mkdir(exist_ok=True)

    data_paths = {
        "cptac_luad": {
            "raw_data": cptac_path,
            "mask_dir": masks_path / "cptac_luad",
            "output_dir": output_basedir / "cptac_luad",
        },
        "cptac_lusc": {
            "raw_data": cptac_path,
            "mask_dir": masks_path / "cptac_lusc",
            "output_dir": output_basedir / "cptac_lusc",
        },
        "tcga_luad": {
            "raw_data": tcga_path,
            "mask_dir": masks_path / "tcga_luad",
            "output_dir": output_basedir / "tcga_luad",
        },
        "tcga_lusc": {
            "raw_data": tcga_path,
            "mask_dir": masks_path / "tcga_lusc",
            "output_dir": output_basedir / "tcga_lusc",
        }
    }

    for key, paths in data_paths.items():
        output_dir = paths["output_dir"]
        output_dir.mkdir(exist_ok=True)
        mask_dir = paths["mask_dir"] / "segments/"
        logger.info(f"Running tiling with superpixels for {key} dataset...")
        tile_dataset(
            paths["raw_data"],
            mask_dir,
            output_dir=output_dir,
            tile_size=224,
            threshold=0.8,
            num_workers_wsi=12,
            num_workers_tiles=4,
            magnification=10,
            save_tile_overlay=True,
            # debug_id="C3N-02155-24",
            average_superpixel_area=486800,
            average_n_tiles=20,
        )

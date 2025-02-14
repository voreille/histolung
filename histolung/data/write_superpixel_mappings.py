import json
from pathlib import Path

from tqdm import tqdm
import pandas as pd

project_dir = Path(__file__).parents[2].resolve()


def generate_superpixel_tile_mapping(
    tile_dir,
    output_json="data/interim/tiles_superpixels/superpixel_mapping.json",
    wsi_ids_to_discard=None,
):
    """
    Generates a JSON file mapping each superpixel to its tile paths as a list of dictionaries.

    Args:
        tile_dir (str or Path): Directory containing the tiles.
        output_json (str): Path to save the JSON mapping.
    """
    tile_dir = Path(tile_dir)
    superpixel_map = {}
    output_json = project_dir / output_json

    # Loop through all tiles and extract the superpixel identifier
    tile_paths = [
        f for f in tqdm(
            tile_dir.rglob("*/tiles/*.png"),
            desc="Gathering all the tile paths and filtering test wsi_ids")
        if f.stem.split("__")[0] not in wsi_ids_to_discard
    ]
    for tile_path in tqdm(tile_paths):
        tile_name = tile_path.stem  # Example: [wsi_id]__l[superpixel_label]__x[x]_y[y]
        superpixel_id = "__".join(
            tile_name.split("__")[:2])  # Extract [wsi_id]__l[superpixel_label]

        if superpixel_id not in superpixel_map:
            superpixel_map[superpixel_id] = []

        superpixel_map[superpixel_id].append(str(tile_path))  # Store full path

    # # Loop through all tiles and extract the superpixel identifier
    # tile_dirs = [
    #     f for f in tqdm(
    #         tile_dir.rglob("*/tiles/"),
    #         desc=
    #         "Gathering all the tile directory paths and filtering test wsi_ids"
    #     ) if f.stem.split("__")[0] not in wsi_ids_to_discard
    # ]
    # for tile_dir in tqdm(tile_dirs):
    #     for tile_path in tile_dir.rglob("*.png"):
    #         tile_name = tile_path.stem  # Example: [wsi_id]__l[superpixel_label]__x[x]_y[y]
    #         superpixel_id = "__".join(tile_name.split(
    #             "__")[:2])  # Extract [wsi_id]__l[superpixel_label]

    #         if superpixel_id not in superpixel_map:
    #             superpixel_map[superpixel_id] = []

    #         superpixel_map[superpixel_id].append(
    #             str(tile_path))  # Store full path

    # Convert to a list of dictionaries for direct indexing
    superpixel_list = [{
        "superpixel_id": key,
        "tile_paths": value
    } for key, value in superpixel_map.items()]

    # Save as JSON
    with open(output_json, "w") as f:
        json.dump(superpixel_list, f, indent=4)

    print(f"Superpixel-tile mapping saved to {output_json}")


if __name__ == "__main__":
    # tile_dir = project_dir / "data/interim/tiles_superpixels/"
    tile_dir = Path(
        "/mnt/nas7/data/Personal/Valentin/histopath/tiles_superpixels_with_overlap/"
    )
    wsi_ids_to_discard = [
        'C3L-01924-25', 'C3L-01924-21', 'C3L-01924-27', 'C3L-01924-23',
        'C3L-01924-22', 'C3L-01924-24', 'C3L-01924-28',
        'TCGA-05-4430-01Z-00-DX1', 'TCGA-05-4415-01Z-00-DX1',
        'TCGA-05-4402-01Z-00-DX1', 'TCGA-05-4427-01Z-00-DX1',
        'TCGA-05-4245-01Z-00-DX1', 'TCGA-05-4422-01Z-00-DX1',
        'TCGA-05-4250-01Z-00-DX1', 'TCGA-05-4398-01Z-00-DX1',
        'TCGA-05-4395-01Z-00-DX1', 'TCGA-05-4418-01Z-00-DX1',
        'TCGA-05-4424-01Z-00-DX1', 'TCGA-05-4434-01Z-00-DX1',
        'TCGA-05-4433-01Z-00-DX1', 'TCGA-05-4397-01Z-00-DX1',
        'TCGA-05-4426-01Z-00-DX1', 'TCGA-05-4432-01Z-00-DX1',
        'TCGA-05-4382-01Z-00-DX1', 'TCGA-05-4244-01Z-00-DX1',
        'TCGA-05-4417-01Z-00-DX1', 'TCGA-05-4403-01Z-00-DX1',
        'TCGA-05-4249-01Z-00-DX1', 'TCGA-05-4420-01Z-00-DX1',
        'TCGA-05-4396-01Z-00-DX1', 'TCGA-05-4405-01Z-00-DX1'
    ]  # these wsis have a mismatch between magnifications and pixel sizes, could resize them but for now just discard

    test_df = pd.read_csv(project_dir / "data/metadata/cptac_test.csv")
    test_wsi_ids = list(test_df["Slide_ID"].values)
    generate_superpixel_tile_mapping(
        tile_dir,
        output_json=tile_dir / "superpixel_mapping_train.json",
        wsi_ids_to_discard=test_wsi_ids + wsi_ids_to_discard,
    )

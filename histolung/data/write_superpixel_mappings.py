import json
from pathlib import Path

from tqdm import tqdm
import pandas as pd

project_dir = Path(__file__).parents[2].resolve()


def generate_superpixel_tile_mapping(
    tile_dir,
    output_json="data/interim/tiles_superpixels/superpixel_mapping.json",
    test_wsi_ids=None,
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
        if f.stem.split("__")[0] not in test_wsi_ids
    ]
    for tile_path in tqdm(tile_paths):
        tile_name = tile_path.stem  # Example: [wsi_id]__l[superpixel_label]__x[x]_y[y]
        superpixel_id = "__".join(
            tile_name.split("__")[:2])  # Extract [wsi_id]__l[superpixel_label]

        if superpixel_id not in superpixel_map:
            superpixel_map[superpixel_id] = []

        superpixel_map[superpixel_id].append(str(tile_path))  # Store full path

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
    tile_dir = project_dir / "data/interim/tiles_superpixels/"
    test_df = pd.read_csv(project_dir / "data/metadata/cptac_test.csv")
    test_wsi_ids = list(test_df["Slide_ID"].values)
    generate_superpixel_tile_mapping(
        tile_dir,
        output_json=
        "data/interim/tiles_superpixels/training_superpixel_tile_map.json",
        test_wsi_ids=test_wsi_ids,
    )

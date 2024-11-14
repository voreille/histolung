from pathlib import Path

from histolung.data.rename import write_wsi_paths_to_csv
from histolung.utils.yaml import load_yaml_with_env

# Base directory and configuration loading
project_dir = Path(__file__).parents[2].resolve()
config_path = project_dir / "histolung/config/datasets_config.yaml"
config = load_yaml_with_env(config_path)
dataset = "tcga_luad"

write_wsi_paths_to_csv(
    "/home/valentin/workspaces/histolung/data/interim/masks/tcga_luad/results.tsv",
    "/home/valentin/workspaces/histolung/data/interim/masks/tcga_luad/test.csv",
    "tcga_luad",
    config["datasets"][dataset],
)

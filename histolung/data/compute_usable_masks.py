import os
from pathlib import Path

from dotenv import load_dotenv

from histolung.data.binarize_with_histoqc import compute_usable_mask_refactor
# Load environment variables from the .env file
load_dotenv()

# Access the variables
TCGA_DATASETS = Path(os.getenv('TCGA_DATA_RAW_PATH'))
DEBUG = True
project_dir = Path(__file__).parents[2].resolve()
if DEBUG:
    DATASETS = {'tcga_lusc': project_dir / "data/debug_histoqc/LUSC"}
else:
    TCGA_DATASETS = Path(os.getenv('TCGA_DATA_PATH')).resolve()
    DATASETS = {
        'tcga_luad': TCGA_DATASETS / "LUSC",
        'tcga_lusc': TCGA_DATASETS / "LUAD",
    }

input_pattern_dict = {
    'tcga_luad': "*/*.svs",
    'tcga_lusc': "*/*.svs",
}

config_path = project_dir / "histolung/config/HistoQC/config.ini"


def main():
    wsi_counter = 0
    for data_name, data_dir in DATASETS.items():
        # wsi_files = list(data_dir.rglob("*.svs"))
        output_dir = project_dir / f"data/interim/masks/{data_name}"
        output_dir.mkdir(exist_ok=True)

        # for wsi_file in wsi_files:
        #     if DEBUG:
        #         if wsi_counter > 1:
        #             break
        # wsi_counter += 1
        compute_usable_mask_refactor(data_dir,
                                     output_dir,
                                     input_pattern="*.svs",
                                     config_path=config_path)


if __name__ == "__main__":
    main()

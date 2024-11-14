from pathlib import Path
from dotenv import load_dotenv

from histolung.data.histoqc import run_histoqc
from histolung.utils.yaml import load_yaml_with_env

load_dotenv()

DEBUG = False

project_dir = Path(__file__).parents[2].resolve()
config = load_yaml_with_env(project_dir /
                            "histolung/config/datasets_config.yaml")

parent_output_dir = project_dir / ("data/interim/debug_masks" if DEBUG else
                                   config["histoqc_masks_basedir"])
parent_output_dir.mkdir(exist_ok=True)


def main():
    for data_name, dataset_info in config["datasets"].items():
        data_dir = Path(dataset_info["data_dir"])
        output_dir = parent_output_dir / data_name
        output_dir.mkdir(exist_ok=True)

        run_histoqc(
            data_dir,
            output_dir,
            input_pattern=dataset_info["input_pattern"],
            config_path=project_dir / dataset_info["config_path"],
            force=False,
            num_workers=12,
        )


if __name__ == "__main__":
    main()

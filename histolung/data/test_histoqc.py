from pathlib import Path

from histolung.data.binarize_with_histoqc import compute_usable_mask


def main():
    input_image = "/mnt/nas6/data/lung_tcga/data/tcga_luad/TCGA-55-7573-01Z-00-DX1.tif"
    output_dir = "/home/valentin/workspaces/histolung/data/test_histoqc"
    config = "/home/valentin/workspaces/histolung/histolung/config/config_histoqc.ini"
    compute_usable_mask(input_image, output_dir, config_path=config)


if __name__ == "__main__":
    main()

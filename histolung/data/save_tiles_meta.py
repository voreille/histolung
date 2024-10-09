import json
from pathlib import Path


def store_wsi_data_to_json(root_dir, labels=None, output_json='wsi_data.json'):
    """
    Store WSI metadata (label, patch directory, patch file paths) into a JSON file.
    
    Args:
        root_dir (str): Path to the root directory containing WSI directories.
        label_map (dict): Mapping from class names to labels.
        output_json (str): Path to the output JSON file.
    """
    if labels is None:
        labels = ["lusc", 'luad']
    wsi_data = []

    # Step 1: Collect WSI labels, patch directory, and patch file paths
    for class_name in labels:
        class_dir = Path(root_dir).resolve() / f'tcga_{class_name}'
        for wsi_id_dir in class_dir.iterdir():
            if wsi_id_dir.is_dir():
                patch_dir = wsi_id_dir / f'{wsi_id_dir.name}_tiles'
                if patch_dir.exists():
                    patch_files = [str(p) for p in patch_dir.glob("*.png")]
                    wsi_data.append({
                        "wsi_id": wsi_id_dir.name,
                        "label": class_name,
                        "patch_dir": str(patch_dir),
                        "patch_files": patch_files,
                    })

    # Step 2: Save the data to a JSON file
    with open(output_json, 'w') as f:
        json.dump(wsi_data, f, indent=4)
    print(f"WSI data saved to {output_json}")


# Example usage:
if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    root_dir = project_dir / 'data/interim'
    store_wsi_data_to_json(
        root_dir,
        labels=["luad", "lusc"],
        output_json=root_dir / 'tcga_wsi_data.json',
    )

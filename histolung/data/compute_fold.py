from pathlib import Path
import json
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd


def generate_folds(json_path, n_splits=5, stratified=True):
    # Load the WSI data from JSON file
    with open(json_path, 'r') as f:
        patches_infos = json.load(f)

    # Prepare the data for the DataFrame
    data = [{
        "wsi_id": d["wsi_id"],
        "label": d["label"]
    } for d in patches_infos]
    df = pd.DataFrame(data)

    # Initialize fold column
    df['fold'] = -1

    # Select the type of KFold split (Stratified or not)
    if stratified:
        skf = StratifiedKFold(n_splits=n_splits)
        X = df['wsi_id']  # Placeholder for input features
        y = df['label']  # The labels (e.g., "lusc" or "luad")

        # Assign fold numbers in a stratified manner
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            df.loc[val_idx, 'fold'] = fold
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        X = df.index  # Use indices for non-stratified split

        # Assign fold numbers in a non-stratified manner
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            df.loc[val_idx, 'fold'] = fold

    # Show fold balance summary
    fold_balance = df.groupby(['fold', 'label']).size().unstack(fill_value=0)
    print("Summary of the repartition of the fold:")
    print(fold_balance)
    return df


def main():
    # Parameters
    project_dir = Path(__file__).resolve().parents[2]
    json_path = project_dir / "data/interim/wsi_data.json"
    output_path = project_dir / "data/interim/tcga_folds.csv"
    n_splits = 5
    stratified = True

    # Generate the folds
    df = generate_folds(json_path, n_splits=n_splits, stratified=stratified)

    df.to_csv(output_path)


if __name__ == "__main__":
    main()

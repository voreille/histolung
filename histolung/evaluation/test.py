from pathlib import Path

import pandas as pd
import numpy as np

from histolung.models.feature_extractor import BaseFeatureExtractor
from histolung.evaluation.datasets import LC25000Dataset, LungHist700Dataset, WSSS4LUADDataset
from histolung.evaluation.evaluators import KNNEvaluator, LinearEvaluator, SegmentationEvaluator
from histolung.evaluation.cross_validator import CrossValidator

project_dir = Path(__file__).parents[2]

if __name__ == "__main__":
    # Example configurations
    datasets = [
        (
            "Patient-Level Dataset",
            LungHist700Dataset,
            project_dir / "data/interim/tiles_LungHist700/metadata.csv",
        ),
    ]

    # Load SSL model
    model = BaseFeatureExtractor.get_feature_extractor(
        "UNI",
        weights_filepath=
        "models/uni/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin",
    )

    for dataset_name, dataset_class, metadata_path in datasets:
        print(f"Evaluating on {dataset_name}")
        metadata_df = pd.read_csv(metadata_path)

        if dataset_name == "Segmentation Dataset":
            evaluator = SegmentationEvaluator(None)
        else:
            evaluator = KNNEvaluator(None, n_neighbors=5)

        dataset = dataset_class(metadata_df)
        validator = CrossValidator(dataset, evaluator)
        scores = validator.cross_validate(model)

        print(f"{dataset_name} CV Scores: {scores}")
        print(f"Mean Score: {np.mean(scores):.4f}")

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from histolung.models.feature_extractor import BaseFeatureExtractor
from histolung.evaluation.datasets import TileDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Replace "0" with the GPU index you want to use

tiles_dir = Path(
    "/home/valentin/workspaces/histolung/data/interim/tiles_LungHist700/tiles/"
)

model = BaseFeatureExtractor.get_feature_extractor(
    "UNI",
    weights_filepath=
    "models/uni/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin",
)


tile_dataset = TileDataset(list(tiles_dir.glob("*.jpg")))
dataloader = DataLoader(tile_dataset, batch_size=32, num_workers=4)

embeddings = []
for batch in dataloader:
    embeddings.append(model(batch))

print(embeddings)
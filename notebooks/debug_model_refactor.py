from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from histolung.legacy.datasets import Dataset_instance_MIL
from histolung.utils import yaml_load
from histolung.models.models import load_pretrained_model, MILModel, AttentionAggregator


def set_dataloader(
    wsi_tiles_path,
    cfg,
    resize_param,
    batch_size=1,
    num_workers=2,
):
    patches = [[str(p)] for p in wsi_tiles_path.glob("*.png")]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.stddev),
        transforms.Resize(size=(resize_param, resize_param), antialias=True),
    ])
    return DataLoader(
        Dataset_instance_MIL(patches, preprocess=preprocess),
        batch_size=batch_size,
        num_workers=num_workers,
    )


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    feature_extractor, feature_dim, resize_param = load_pretrained_model(
        "resnet50")

    wsi_tiles_path = Path(
        "/home/valentin/workspaces/histolung/data/test_heatmaps/TCGA-18-3417-01Z-00-DX1/TCGA-18-3417-01Z-00-DX1_tiles"
    )
    cfg_path = "/home/valentin/workspaces/histolung/models/MIL/first/config.yml"
    cfg = yaml_load(cfg_path)

    dataloader = set_dataloader(wsi_tiles_path, cfg, resize_param)

    mil_model = MILModel.from_config(cfg_path).to(device)

    pred, attention = mil_model.process_dataloader(dataloader)

from pathlib import Path

import torch
import torch.nn as nn

from histolung.models.models_darya import MoCoV2Encoder
from histolung.evaluation.evaluators import LungHist700Evaluator


def load_model(checkpoint_path, device):
    """Load the MoCoV2 model from a given checkpoint, handling missing keys like queue_ptr."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Checkpoint {checkpoint_path} not found, skipping...")
        return None, None

    model = MoCoV2Encoder()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load the model state dict with strict=False to ignore missing keys (like queue_ptr)
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False)

    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

    encoder = model.encoder_q
    encoder.fc = nn.Identity()

    return encoder.to(device).eval()


def main():
    evaluator = LungHist700Evaluator(
        data_dir=
        "/home/valentin/workspaces/histolung/data/processed/LungHist700_10x",
        batch_size=256,
        num_workers=12,
        gpu_id=0,
        preprocess=None,  # if None apply the standard one 
        n_splits=5,  # LPO for Leave-One-Patient out
        seed=42,  # for reproducibility of the fold
        magnification=
        "all",  # "all", "20x", or "40x" to select only the LungHist700 patches for those magnifications
        aggregate=True,  # To aggregate the embeddings (average by default)
        n_neighbors=5,  # Number of Neigbors in the kNN classifier
    )
    model = load_model(
        "/mnt/nas7/data/Personal/Darya/saved_models/superpixel_org/superpixel_org_99.pth",
        evaluator.device,
    )

    embeddings, tile_ids = evaluator.compute_embeddings(model)
    aggregated_embeddings, image_ids, labels, patient_ids = evaluator.aggregate_embeddings(
        embeddings, tile_ids)
    # Then do wathever you want like evaluating kNN
    results = evaluator.evaluate(embeddings, tile_ids)
    print(results)


if __name__ == "__main__":
    main()

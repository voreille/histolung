import torch
import torch.nn as nn
from pathlib import Path
from histolung.models.models_darya import MoCoV2Encoder


def get_device(gpu_id=None):
    """Select the appropriate device for computation."""
    if torch.cuda.is_available():
        if gpu_id is not None and gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda:0")  # Default to first GPU
    else:
        device = torch.device("cpu")
    print(f"Using {device}")
    return device


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


def compare_checkpoints(checkpoint1, checkpoint2, device="cpu"):
    """Load two checkpoints and compare the weights of the first layer."""

    # Load models from checkpoints
    model1 = load_model(checkpoint1, device)
    model2 = load_model(checkpoint2, device)

    if model1 is None or model2 is None:
        print("One or both models could not be loaded.")
        return

    # Extract first convolutional layer weights
    weights1 = model1.conv1.weight
    weights2 = model2.conv1.weight

    # Compute the absolute difference
    weight_diff = torch.abs(weights1 - weights2)

    # Print summary statistics
    print(f"Max difference: {weight_diff.max().item()}")
    print(f"Mean difference: {weight_diff.mean().item()}")
    print(
        f"Non-zero differences: {(weight_diff > 0).sum().item()} out of {weight_diff.numel()}"
    )

    # Check if weights are identical
    if torch.allclose(weights1, weights2, atol=1e-6):
        print("Weights are identical within tolerance.")
    else:
        print("Weights are different.")


if __name__ == "__main__":
    checkpoint1 = "/mnt/nas7/data/Personal/Darya/saved_models/superpixels_moco_org/superpixel_moco_org_0.pth"
    checkpoint2 = "/mnt/nas7/data/Personal/Darya/saved_models/superpixels_moco_org/superpixel_moco_org_58.pth"
    device = get_device(gpu_id=1)
    compare_checkpoints(checkpoint1, checkpoint2, device=device)

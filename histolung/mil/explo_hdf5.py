import h5py
import torch
from pathlib import Path

# Example function to simulate embedding generation
def generate_fake_embeddings(wsi_id, embedding_dim=512, num_tiles=100):
    """
    Generates fake embeddings for a given WSI.
    
    Args:
        wsi_id (str): WSI ID.
        embedding_dim (int): Dimension of each embedding vector.
        num_tiles (int): Number of tiles for the WSI.
        
    Returns:
        torch.Tensor: Randomly generated embeddings for the WSI.
    """
    return torch.randn(num_tiles, embedding_dim)

def save_embeddings_to_hdf5(hdf5_file_path, embeddings_dict):
    """
    Saves embeddings to an HDF5 file, with each WSI stored as a separate dataset.
    
    Args:
        hdf5_file_path (str or Path): Path to the HDF5 file.
        embeddings_dict (dict): Dictionary where keys are WSI IDs and values are embeddings (torch.Tensor).
    """
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        embeddings_group = hdf5_file.create_group('embeddings')  # Optional group

        for wsi_id, embeddings in embeddings_dict.items():
            # Store embeddings for each WSI under its wsi_id
            embeddings_group.create_dataset(wsi_id, data=embeddings.numpy())

    print(f"Embeddings successfully saved to {hdf5_file_path}")

# Main function to test storing embeddings
def main():
    # Define HDF5 file path
    hdf5_file_path = Path('wsi_embeddings.h5')

    # Generate some fake embeddings for testing
    wsi_embeddings = {
        'wsi_001': generate_fake_embeddings('wsi_001'),
        'wsi_002': generate_fake_embeddings('wsi_002'),
        'wsi_003': generate_fake_embeddings('wsi_003'),
    }

    # Save the embeddings to an HDF5 file
    save_embeddings_to_hdf5(hdf5_file_path, wsi_embeddings)

if __name__ == "__main__":
    main()

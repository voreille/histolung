import click
import subprocess
from pathlib import Path

# Define the URLs for the files
TAR_GZ_URL = "https://zenodo.org/records/7412731/files/panCK_Epithelium.tar.gz?download=1"
CSV_URL = "https://zenodo.org/records/7412731/files/panCK_fileinfo.csv?download=1"

# Determine the project directory dynamically (two levels above this script's directory)
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Define the target directory relative to the project directory
TARGET_DIR = PROJECT_DIR / "data/raw/SegPath/panCK"
TMP_DIR = PROJECT_DIR / "data/raw/SegPath/.tmp"  # Hidden directory for tar files

@click.command()
def download_and_extract():
    """Download the .tar.gz and .csv files using wget, and extract them to {project_dir}/data/raw/SegPath/panCK."""

    # Ensure the target and temporary directories exist
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Check if data has already been downloaded and extracted
    if is_data_downloaded():
        click.echo("Data already downloaded and extracted. Skipping download.")
        return

    # Download and extract the .tar.gz file using wget
    download_with_wget(TAR_GZ_URL, TMP_DIR / "panCK_Epithelium.tar.gz")

    # Extract the tar.gz file
    extract_tar_gz(TMP_DIR / "panCK_Epithelium.tar.gz", TARGET_DIR)

    # Download the CSV file using wget
    download_with_wget(CSV_URL, TARGET_DIR / "panCK_fileinfo.csv")

    click.echo("Download and extraction complete.")

def is_data_downloaded():
    """Check if the data has already been downloaded and extracted."""
    # Check if the extracted directory already contains the expected files
    expected_files = ["panCK_fileinfo.csv"]  # Add more expected files if needed
    for file in expected_files:
        if not (TARGET_DIR / file).exists():
            return False
    return True

def download_with_wget(url, output_path):
    """Download a file using wget."""
    click.echo(f"Downloading {output_path.name} with wget...")
    subprocess.run(['wget', '-O', str(output_path), url], check=True)

def extract_tar_gz(tar_gz_path, extract_to_dir):
    """Extract a .tar.gz file to the specified directory."""
    click.echo(f"Extracting {tar_gz_path.name}...")
    subprocess.run(['tar', '-xzf', str(tar_gz_path), '-C', str(extract_to_dir)], check=True)

    # Optionally delete the tar.gz file after extraction
    tar_gz_path.unlink()
    click.echo(f"{tar_gz_path.name} deleted after extraction.")

if __name__ == "__main__":
    download_and_extract()

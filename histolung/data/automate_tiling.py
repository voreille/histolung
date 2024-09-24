import os
from pathlib import Path
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from histolung.data.tile_image import tile_image

# Load environment variables from .env file
load_dotenv()

# Define datasets in the .env file
DEBUG = False
project_dir = Path(__file__).parents[2].resolve()
if DEBUG:
    DATASETS = {'tcga_luad': project_dir / "data/test_raw/tcga_luad"}

else:
    TCGA_DATASETS = Path(os.getenv('TCGA_DATA_PATH')).resolve()
    DATASETS = {
        'tcga_luad': TCGA_DATASETS / "tcga_luad",
        'tcga_lusc': TCGA_DATASETS / "tcga_lusc",
    }


def setup_logging(dataset_name):
    """
    Set up logging for both the console and a log file specific to the dataset.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = project_dir / "logs/data/"
    logs_dir.mkdir(exist_ok=True, parents=True)
    log_filename = logs_dir / f"tiling_{dataset_name}_{timestamp}.log"

    logger = logging.getLogger(
        "histolung.data.tile_image")  # Reference tile_image.py logger
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'))
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized for dataset {dataset_name}.")
    return logger


def process_dataset(dataset_name, dataset_path, docker_image, logger):
    """
    Process the images in the given dataset directory using multithreading for Docker processes.
    """

    if DEBUG:
        output_dir = project_dir / f"data/test_interim/{dataset_name}"
        max_workers = 1
    else:
        max_workers = 12
        output_dir = project_dir / f"data/interim/{dataset_name}"

    output_dir.mkdir(
        parents=True,
        exist_ok=True)  # Create the output directory if it doesn't exist

    # Collect all .tif files using Path
    image_files = list(dataset_path.glob("*"))

    logger.info(
        f"Processing dataset '{dataset_name}' with {len(image_files)} images.")

    # Using ThreadPoolExecutor for multithreading, handling external Docker processes
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(tile_image, str(image_file), str(output_dir),
                            docker_image) for image_file in image_files
        ]

    for future in futures:
        try:
            future.result(
            )  # This will raise an exception if a thread encounters one
        except Exception as e:
            logger.error(f"Error in processing: {e}")


# Loop through datasets and process each sequentially
docker_image = 'mmunozag/pyhist'

for dataset_name, dataset_path in DATASETS.items():
    if dataset_path and dataset_path.exists():
        logger = setup_logging(dataset_name)
        logger.info(f"Processing dataset: {dataset_name} at {dataset_path}")
        process_dataset(dataset_name, dataset_path, docker_image, logger)
    else:
        logger = logging.getLogger('default')
        logger.warning(
            f"Dataset path for {dataset_name} not found or does not exist.")

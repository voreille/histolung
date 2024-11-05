import os
import re
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# A flag to keep track if the Docker image has been checked
docker_image_checked = False


def check_and_pull_docker_image(docker_image):
    """
    Check if the Docker image is present, and pull it if it's not.
    """
    global docker_image_checked  # Use global to ensure the check is shared across calls

    if docker_image_checked:
        return

    try:
        subprocess.run(["docker", "inspect", "--type=image", docker_image],
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)
        logger.info(f"Docker image {docker_image} is already present.")
    except subprocess.CalledProcessError:
        logger.info(
            f"Docker image {docker_image} not found. Pulling the image...")
        try:
            subprocess.run(["docker", "pull", docker_image], check=True)
            logger.info(f"Successfully pulled Docker image: {docker_image}")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to pull Docker image {docker_image}. Error: {e}")
            raise

    # Set flag to True after the first successful check
    docker_image_checked = True


def parse_and_log_output(output: str, logger: logging.Logger, context: str):
    """
    Parse the output of a docker command and log messages at appropriate log levels.

    Args:
        output (str): The output from the subprocess.
        logger (logging.Logger): The logger to log the messages.
        context (str): Context for the logging (e.g., the image name or command being run).
    """
    if output:
        for line in output.splitlines():
            # Use regex to extract the log level and the actual message
            match = re.match(r'.*\[(\w+)\]:\s*(.+)', line)
            if match:
                log_level = match.group(
                    1).strip()  # Extract the log level part
                message = match.group(2).strip()  # Extract the message part
                if log_level == "CRITICAL":
                    logger.critical(f"{context}: {message}")
                elif log_level == "ERROR":
                    logger.error(f"{context}: {message}")
                elif log_level == "WARNING":
                    logger.warning(f"{context}: {message}")
                elif log_level == "INFO":
                    logger.info(f"{context}: {message}")
                elif log_level == "DEBUG":
                    logger.debug(f"{context}: {message}")
                else:
                    # Default to info if no log level is detected
                    logger.info(f"{context}: {message}")
            else:
                # Log the full line if it doesn't match the expected pattern
                logger.info(f"{context}: {line}")


def compute_usable_mask(
    input_image,
    output_dir,
    docker_image='histotools/histoqc:master',
    user=None,
    config_path=None,
):
    """
    Tile a single image using HistoQC Docker, capturing and logging Docker stdout and stderr.
    """
    # Ensure Docker image is present (check only once)
    check_and_pull_docker_image(docker_image)

    # Default user if not specified
    user = user or f'{os.getuid()}:{os.getgid()}'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    input_image = Path(input_image)
    output_dir = Path(output_dir)

    # Ensure the config path is provided
    if config_path is not None:
        config_path = Path(config_path)
    else:
        raise ValueError("Config path must be provided for HistoQC.")

    command = [
        "docker", "run", "--rm", "-v", f"{input_image.parent}:/data_ro:ro",
        "-v", f"{output_dir}:/data", "-v",
        f"{config_path.parent}:{config_path.parent}", "--name",
        f"histoqc_{input_image.stem}", "-u", user, docker_image, "/bin/bash",
        "-c",
        f"histoqc_pipeline /data_ro/{input_image.name} -o /data/output -c {config_path}"
    ]

    try:
        # Capture stdout and stderr from the Docker process
        result = subprocess.run(command,
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)

        # Log the Docker stdout and stderr
        context = f"HistoQC output for '{input_image.name}'"
        parse_and_log_output(result.stdout, logger, context)
        parse_and_log_output(result.stderr, logger, context)

        logger.info(f"Tiling completed for image {input_image.name}.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error during tiling for {input_image.name}: {e}")
        raise


def compute_usable_mask_refactor(
    input_dir,
    output_dir,
    input_pattern="*.svs",
    docker_image='histotools/histoqc:master',
    user=None,
    config_path=None,
):
    """
    Tile a single image using HistoQC Docker, capturing and logging Docker stdout and stderr.
    """
    # Ensure Docker image is present (check only once)
    check_and_pull_docker_image(docker_image)

    # Default user if not specified
    user = user or f'{os.getuid()}:{os.getgid()}'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Ensure the config path is provided
    if config_path is not None:
        config_path = Path(config_path)
    else:
        raise ValueError("Config path must be provided for HistoQC.")

    command = [
        "docker", "run", "--rm", "-v", f"{input_dir}:/data_ro:ro", "-v",
        f"{output_dir}:/data", "-v",
        f"{config_path.parent}:{config_path.parent}", "--name",
        f"histoqc_{input_dir.name}", "-u", user, docker_image, "/bin/bash",
        "-c",
        f"histoqc_pipeline /data_ro/{input_pattern} -o /data/output -c {config_path}"
    ]

    try:
        # Capture stdout and stderr from the Docker process
        result = subprocess.run(command, check=True)

        # Log the Docker stdout and stderr
        context = f"HistoQC output for '{input_dir.name}'"
        parse_and_log_output(result.stdout, logger, context)
        parse_and_log_output(result.stderr, logger, context)

        logger.info(f"Tiling completed for image {input_dir.name}.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error during tiling for {input_dir.name}: {e}")
        raise

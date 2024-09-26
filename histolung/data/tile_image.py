import os
import re
import subprocess
from pathlib import Path
import logging
from threading import current_thread

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
    Parse the output of pyhist docker command and log messages at appropriate log levels.

    Args:
        stderr (str): The stderr output from the subprocess.
        logger (logging.Logger): The logger to log the messages.
        context (str): Context for the logging (e.g., the image name or command being run).
    """
    if output:
        for line in output.splitlines():
            # Use regex to extract the log level and the actual message
            match = re.match(r'.*\[(\w+)\]:\s*(.+)', line)
            if match:
                log_level = match.group(1).strip()  # Extract the loglevel part
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


def tile_image(
    input_image,
    output_dir,
    docker_image='mmunozag/pyhist',
    user=None,
    downsample=1,
    mask_downsample=8,
    crossed_image=False,
):
    """
    Tile a single image using Docker, capturing and logging Docker stdout and stderr.
    """
    # Ensure Docker image is present (check only once)
    check_and_pull_docker_image(docker_image)

    # Default user if not specified
    user = user or f'{os.getuid()}:{os.getgid()}'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    input_image = Path(input_image)
    command = [
        "docker", "run", "--rm", "-v", f"{input_image.parent}:/pyhist/images/",
        "-v", f"{os.path.abspath(output_dir)}:/pyhist/output/", "-u", user,
        "-v", "/etc/passwd:/etc/passwd", docker_image, "--method", "graph",
        "--mask-downsample",
        str(mask_downsample), "--output-downsample",
        str(downsample), "--tilecross-downsample", "32", "--corners", "1111",
        "--borders", "0000", "--percentage-bc", "1", "--k-const", "1000",
        "--minimum_segmentsize", "1000", "--info", "verbose",
        "--content-threshold", "0.2", "--patch-size", "256", "--save-patches",
        "--save-mask", "--save-tilecrossed-image", "--output",
        "/pyhist/output/", f"/pyhist/images/{input_image.name}"
    ]

    try:
        # Capture stdout and stderr from the Docker process
        result = subprocess.run(command,
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)

        # Log the Docker stdout
        context = f"PyHIST output for '{input_image.name}'"
        parse_and_log_output(result.stdout, logger, context)
        # Log any Docker stderr as well
        parse_and_log_output(result.stderr, logger, context)

        logger.info(f"Tiling completed for image {input_image.name}.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error during tiling for {input_image.name}: {e}")
        raise

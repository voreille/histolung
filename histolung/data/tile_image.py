import os
import subprocess

import click
import mlflow


def check_and_pull_docker_image(docker_image):
    """
    Check if the Docker image is present, and pull it if it's not.
    """
    try:
        # Check if the Docker image is present
        subprocess.run(["docker", "inspect", "--type=image", docker_image],
                       check=True,
                       stdout=subprocess.DEVNULL)
        print(f"Docker image {docker_image} is already present.")
    except subprocess.CalledProcessError:
        # Image is not present, so pull it
        print(f"Docker image {docker_image} not found. Pulling the image...")
        try:
            subprocess.run(["docker", "pull", docker_image], check=True)
            print(f"Successfully pulled Docker image: {docker_image}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to pull Docker image {docker_image}. Error: {e}")
            raise


@click.command()
@click.option(
    '--input_image',
    '-i',
    #   required=True,
    default=
    "/home/val/workspaces/histolung/data/test_raw/TCGA-4B-A93V-01Z-00-DX1.tif",
    help='Path to the directory containing the input images.')
@click.option(
    '--output_dir',
    '-o',
    # required=True,
    default="/home/val/workspaces/histolung/data/test_interim",
    help='Path to the directory where the output images will be saved.')
@click.option('--docker_image',
              '-d',
              default='mmunozag/pyhist',
              help='Docker image to use for tiling the images.')
@click.option(
    '--user',
    '-u',
    default=f'{os.getuid()}:{os.getgid()}',
    help='User ID and Group ID to pass to Docker for correct permissions.')
@click.option('--crossed_image',
              is_flag=True,
              help='Whether to save the tile-crossed image.')
def tile_images(input_image, output_dir, docker_image, user, crossed_image):
    """
    A CLI tool to tile images using the Docker container.
    """
    input_image = input_image.split("/")
    input_dir = "/".join(input_image[:-1])
    filename = input_image[-1]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check and pull the Docker image if needed
    check_and_pull_docker_image(docker_image)

    # Construct the Docker command
    command = [
        "docker", "run", "-v", f"{os.path.abspath(input_dir)}:/pyhist/images/",
        "-v", f"{os.path.abspath(output_dir)}:/pyhist/output/", "-u", user,
        "-v", "/etc/passwd:/etc/passwd", docker_image, "--method", "graph",
        "--mask-downsample", "8", "--output-downsample", "2",
        "--tilecross-downsample", "32", "--corners", "1111", "--borders",
        "0000", "--percentage-bc", "1", "--k-const", "1000",
        "--minimum_segmentsize", "1000", "--info", "verbose",
        "--content-threshold", "0.2", "--patch-size", "256", "--save-patches",
        "--save-mask", "--save-tilecrossed-image", "--output",
        "/pyhist/output/", f"/pyhist/images/{filename}"
    ]

    # # Add optional arguments
    # if crossed_image:
    #     command.append("--save-tilecrossed-image")

    # # Specify the input image path and the output path in the container
    # input_image_path = f"/pyhist/images/"
    # output_path = f"--output /pyhist/output/ {input_image_path}"

    # # Append output argument and image directory
    # command.extend(output_path.split())

    # Run the Docker command
    try:
        print("Running the Docker tiling process...")
        result = subprocess.run(command, check=True)
        print("Image tiling completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during tiling: {e}")


if __name__ == "__main__":
    tile_images()

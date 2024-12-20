from pathlib import Path
from PIL import Image
import os

from dotenv import load_dotenv

load_dotenv()

# Define the folder containing the images and the output folder
input_folder = Path(os.getenv("LUNGHIST700_RAW"))
output_folder = Path("path/to/output/folder")
output_folder.mkdir(parents=True, exist_ok=True)

# Define the tile size
tile_size = 224
tile_magnification = 10.0

# Iterate through all .jpg files in the input folder
for file in input_folder.rglob("*.jpg"):
    # Extract filename components
    filename = file.stem
    parts = filename.split("_")
    if len(parts) == 3 and parts[0] == "nor":
        cancer_type, magnification, image_id = parts
        differentiation = None
    elif len(parts) == 4:
        cancer_type, differentiation, magnification, image_id = parts
    else:
        print(f"Skipping unexpected file format: {file}")
        continue

    # Extract magnification and determine downsampling factor
    magnification = float(magnification.replace("x", ""))
    downsampling_factor = tile_magnification / magnification

    # Open the image and downsample if needed
    with Image.open(file) as img:
        if downsampling_factor < 1:
            new_width = int(img.width * downsampling_factor)
            new_height = int(img.height * downsampling_factor)
            img = img.resize((new_width, new_height), Image.LANCSZOS)

        # Calculate the number of tiles
        num_tiles_x = img.width // tile_size
        num_tiles_y = img.height // tile_size

        # Generate tiles
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                # Crop the tile
                left = i * tile_size
                top = j * tile_size
                right = left + tile_size
                bottom = top + tile_size
                tile = img.crop((left, top, right, bottom))

                # Create the new filename
                tile_id = f"tile_{i}_{j}"
                new_filename = f"{filename}_{tile_id}.jpg"
                tile_output_path = output_folder / new_filename

                # Save the tile
                tile.save(tile_output_path, "JPEG")

print(f"Tiling complete! Tiles saved to {output_folder}")

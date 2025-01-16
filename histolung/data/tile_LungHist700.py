from pathlib import Path
from PIL import Image, ImageDraw
import os
from dotenv import load_dotenv

load_dotenv()

# Define the folder containing the images and the output folder
project_dir = Path(__file__).parents[2].resolve()
input_folder = Path(os.getenv("LUNGHIST700_RAW"))
output_folder = project_dir / "data/interim/tiles_LungHist700"
outline_folder = output_folder / "outline"
output_folder.mkdir(parents=True, exist_ok=True)
outline_folder.mkdir(parents=True, exist_ok=True)

# Define the tile size and stride
tile_size = 224
tile_stride = 112  # Overlap of 50%

# Iterate through all .jpg files in the input folder
for file in input_folder.rglob("*.jpg"):
    filename = file.stem

    with Image.open(file) as img:
        # Resize the image to the nearest multiple of the tile size
        new_width = ((img.width // tile_stride) + 1) * tile_stride
        new_height = ((img.height // tile_stride) + 1) * tile_stride
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a copy of the image for drawing tile outlines
        outlined_image = img.copy()
        draw = ImageDraw.Draw(outlined_image)

        # Generate tiles
        num_tiles_x = (img.width - tile_size) // tile_stride + 1
        num_tiles_y = (img.height - tile_size) // tile_stride + 1

        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                # Compute tile coordinates
                left = i * tile_stride
                top = j * tile_stride
                right = left + tile_size
                bottom = top + tile_size

                # Crop the tile
                tile = img.crop((left, top, right, bottom))

                # Optionally filter blank tiles (e.g., all white/black tiles)
                if tile.getbbox() is None:  # Check if tile is blank
                    continue

                # Save the tile
                tile_id = f"tile_{i}_{j}"
                tile_output_path = output_folder / f"{filename}_{tile_id}.jpg"
                tile.save(tile_output_path, "JPEG")

                # Draw the outline on the copy
                draw.rectangle([left, top, right, bottom],
                               outline="red",
                               width=2)

        # Save the outlined image
        outlined_image_path = outline_folder / f"{filename}_outlined.jpg"
        outlined_image.save(outlined_image_path, "JPEG")

print(
    f"Tiling complete! Tiles saved to {output_folder} and outlined images to {outline_folder}"
)

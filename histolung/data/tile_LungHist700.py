from pathlib import Path
import re
import os

from PIL import Image, ImageDraw
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
import pandas as pd

load_dotenv()

# Define the folder containing the images and the output folder
project_dir = Path(__file__).parents[2].resolve()
input_folder = Path(os.getenv("LUNGHIST700_RAW"))
output_folder = project_dir / "data/processed/LungHist700"
tiles_folder = output_folder / "tiles"
outline_folder = output_folder / "outline"
tiles_folder.mkdir(parents=True, exist_ok=True)
outline_folder.mkdir(parents=True, exist_ok=True)

# Define tile size and desired magnification
tile_size = 224
desired_magnification = 10
tile_dataset = True

# Define border styles
border_styles = [
    {
        "color": "red",
        "width": 2,
        "style": "solid"
    },  # Solid red
    {
        "color": "blue",
        "width": 2,
        "style": "dotted"
    },  # Dotted blue
    {
        "color": "green",
        "width": 3,
        "style": "dashed"
    },  # Dashed green
    {
        "color": "yellow",
        "width": 2,
        "style": "solid"
    },  # Solid yellow
    {
        "color": "purple",
        "width": 3,
        "style": "dotted"
    },  # Dotted purple
]


def extract_magnification(filename):
    """Extract magnification from filename."""
    match = re.search(r"_([24]0)x_", filename)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Magnification not found in filename: {filename}")


def calculate_stride(image_dim, tile_size):
    """Calculate stride to align tiles symmetrically with borders."""
    if image_dim <= tile_size:
        raise ValueError("Image dimension must be larger than tile size")

    n_tiles = np.ceil(image_dim / tile_size)
    if n_tiles == 1:
        return 0  # Single tile, no stride needed
    total_stride_space = image_dim - tile_size * n_tiles
    stride = tile_size + total_stride_space // (n_tiles - 1)
    return int(stride)


def draw_styled_border(draw, left, top, right, bottom, style):
    """Draw styled borders (solid, dotted, dashed) on the image."""
    color = style["color"]
    width = style["width"]
    border_style = style["style"]

    if border_style == "solid":
        for w in range(width):
            draw.rectangle([left + w, top + w, right - w, bottom - w],
                           outline=color)

    elif border_style == "dotted":
        step = 5  # Distance between dots
        for w in range(width):
            # Top border
            for x in range(left + w, right - w, step):
                draw.point((x, top + w), fill=color)
            # Bottom border
            for x in range(left + w, right - w, step):
                draw.point((x, bottom - w), fill=color)
            # Left border
            for y in range(top + w, bottom - w, step):
                draw.point((left + w, y), fill=color)
            # Right border
            for y in range(top + w, bottom - w, step):
                draw.point((right - w, y), fill=color)

    elif border_style == "dashed":
        dash_length = 10  # Length of dashes
        space_length = 5  # Space between dashes
        for w in range(width):
            # Top border
            for x in range(left + w, right - w, dash_length + space_length):
                draw.line([x, top + w, x + dash_length, top + w],
                          fill=color,
                          width=1)
            # Bottom border
            for x in range(left + w, right - w, dash_length + space_length):
                draw.line([x, bottom - w, x + dash_length, bottom - w],
                          fill=color,
                          width=1)
            # Left border
            for y in range(top + w, bottom - w, dash_length + space_length):
                draw.line([left + w, y, left + w, y + dash_length],
                          fill=color,
                          width=1)
            # Right border
            for y in range(top + w, bottom - w, dash_length + space_length):
                draw.line([right - w, y, right - w, y + dash_length],
                          fill=color,
                          width=1)


files = [f for f in input_folder.rglob("*.jpg")]

if tile_dataset:
    for file in tqdm(files):
        filename = file.stem
        magnification = extract_magnification(filename)
        resampling_factor = desired_magnification / magnification

        with Image.open(file) as img:
            # Resize the image based on desired magnification
            new_height = int(img.height * resampling_factor)
            new_width = int(img.width * resampling_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Calculate strides for symmetrical tiling
            stride_x = calculate_stride(new_width, tile_size)
            stride_y = calculate_stride(new_height, tile_size)

            # Create a copy of the image for drawing tile overlays
            outlined_image = img.convert("RGBA")  # Add alpha channel
            overlay = Image.new("RGBA", outlined_image.size,
                                (255, 255, 255, 0))  # Transparent overlay
            draw = ImageDraw.Draw(overlay)

            # Generate tiles
            x_positions = list(range(0, new_width - tile_size + 1, stride_x))
            y_positions = list(range(0, new_height - tile_size + 1, stride_y))

            for i, left in enumerate(x_positions):
                for j, top in enumerate(y_positions):
                    right = left + tile_size
                    bottom = top + tile_size

                    # Crop the tile
                    tile = img.crop((left, top, right, bottom))

                    # Optionally filter blank tiles (e.g., all white/black tiles)
                    if tile.getbbox() is None:  # Check if tile is blank
                        continue

                    # Save the tile
                    tile_id = f"tile_{i}_{j}"
                    tile_output_path = tiles_folder / f"{filename}_{tile_id}.png"
                    tile.save(tile_output_path, "PNG")

                    # Choose border style based on row/column
                    style_index = (i + j) % len(border_styles)
                    style = border_styles[style_index]

                    # Draw styled border
                    draw_styled_border(draw, left, top, right, bottom, style)

            # Merge the overlay with the original image
            outlined_image = Image.alpha_composite(outlined_image, overlay)

            # Save the outlined image
            outlined_image = outlined_image.convert(
                "RGB")  # Convert back to RGB for saving
            outlined_image_path = outline_folder / f"{filename}_outlined.jpg"
            outlined_image.save(outlined_image_path, "JPEG")

    print(
        f"Tiling complete! Tiles saved to {tiles_folder} and outlined images to {outline_folder}"
    )

# generate the metadata.csv file


def load_metadata():
    label_mapping = {
        "aca_bd": 0,
        "aca_md": 1,
        "aca_pd": 2,
        "nor": 3,
        "scc_bd": 4,
        "scc_md": 5,
        "scc_pd": 6,
    }
    metadata = pd.read_csv(input_folder / "data/data.csv")

    metadata['filename'] = metadata.apply(
        lambda row: "_".join([
            str(row[col])
            for col in ['superclass', 'subclass', 'resolution', 'image_id']
            if pd.notna(row[col])
        ]),
        axis=1,
    )
    metadata['class_name'] = metadata.apply(
        lambda row: f"{row['superclass']}_{row['subclass']}"
        if pd.notna(row['subclass']) else row['superclass'],
        axis=1,
    )

    metadata['label'] = metadata['class_name'].map(label_mapping)
    return metadata


metadata = load_metadata()

tiles_path = [f for f in tiles_folder.glob("*.png")]

# List to store intermediate DataFrames
rows = []

for tile_path in tiles_path:
    original_filename = tile_path.stem.split("_tile_")[0]
    matching_row = metadata[metadata['filename'] == original_filename]

    if not matching_row.empty:
        row = matching_row.iloc[0]
        rows.append({
            'tile_id': tile_path.stem,
            'patient_id': row['patient_id'],
            'superclass': row['superclass'],
            'subclass': row['subclass'],
            'resolution': row['resolution'],
            'image_id': row['image_id'],
            'class_name': row['class_name'],
            'label': row['label'],
            'original_filename': original_filename,
            'tile_path': str(tile_path),
        })

# Save to a CSV file
pd.DataFrame(rows).to_csv(output_folder / "metadata.csv", index=False)

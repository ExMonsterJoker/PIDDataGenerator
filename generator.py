import os
import json
import random
import itertools
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import shutil


def load_config(config_path='config/config.json'):
    """Loads the configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_icons_by_class(icons_dir):
    """Recursively finds all PNG icons and groups them by class."""
    icons_by_class = {}
    for root, _, files in os.walk(icons_dir):
        if not files:
            continue
        class_name = os.path.basename(root)
        if class_name not in icons_by_class:
            icons_by_class[class_name] = []

        for file in files:
            if file.lower().endswith('.png'):
                icons_by_class[class_name].append(os.path.join(root, file))

    class_names = sorted(icons_by_class.keys())
    return icons_by_class, class_names


def generate_dataset(config):
    """Generates the synthetic dataset based on the configuration."""
    icons_dir = config['input']['icons_dir']
    output_dir = config['output']['dir']
    img_width, img_height = config['output']['image_size']
    num_images = config['output']['num_images']
    bg_color = tuple(config['input']['background_color'])
    min_objects = config['generation']['objects_per_image']['min']
    max_objects = config['generation']['objects_per_image']['max']
    allow_overlap = config['generation']['allow_overlap']
    aug_config = config.get('augmentation', {})  # Get augmentation config safely

    # Setup output directories
    images_out_dir = os.path.join(output_dir, 'images')
    labels_out_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    icons_by_class, class_names = get_icons_by_class(icons_dir)

    if not class_names:
        print("Error: No icon classes found. Check the 'icons_dir' in your config.")
        return

    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    class_to_id = {name: i for i, name in enumerate(class_names)}
    class_cycler = itertools.cycle(class_names)

    print(f"Generating {num_images} images with P&ID-specific augmentations...")
    for i in tqdm(range(num_images)):
        background = Image.new('RGB', (img_width, img_height), bg_color)
        num_objects = random.randint(min_objects, max_objects)
        placed_boxes = []
        annotations = []

        for _ in range(num_objects):
            current_class = next(class_cycler)
            if not icons_by_class[current_class]:
                continue
            icon_path = random.choice(icons_by_class[current_class])

            icon = Image.open(icon_path).convert("RGBA")

            # --- New Augmentation Logic ---
            if aug_config.get('enabled', False):
                # Scaling
                scale_config = aug_config.get('scale', {'min': 1.0, 'max': 1.0})
                scale = random.uniform(scale_config['min'], scale_config['max'])
                if scale != 1.0:
                    icon = icon.resize((int(icon.width * scale), int(icon.height * scale)), Image.Resampling.LANCZOS)

                # Discrete Rotation (0, 90, 180, 270 degrees)
                if aug_config.get('rotate_90_degrees', False):
                    angle = random.choice([0, 90, 180, 270])
                    if angle != 0:
                        icon = icon.rotate(angle, expand=True)

                # Horizontal Flip (50% chance)
                if aug_config.get('flip_horizontal', False) and random.random() > 0.5:
                    icon = icon.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

                # Vertical Flip (50% chance)
                if aug_config.get('flip_vertical', False) and random.random() > 0.5:
                    icon = icon.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            # --- End of Augmentation Logic ---

            for _ in range(10):
                # Ensure placement is possible after potential rotation
                if icon.width > img_width or icon.height > img_height:
                    break  # Skip if icon is too large for the canvas

                x = random.randint(0, img_width - icon.width)
                y = random.randint(0, img_height - icon.height)

                new_box = (x, y, x + icon.width, y + icon.height)

                if not allow_overlap:
                    is_overlapping = False
                    for placed_box in placed_boxes:
                        if (new_box[0] < placed_box[2] and new_box[2] > placed_box[0] and
                                new_box[1] < placed_box[3] and new_box[3] > placed_box[1]):
                            is_overlapping = True
                            break
                    if is_overlapping:
                        continue

                background.paste(icon, (x, y), icon)
                placed_boxes.append(new_box)

                class_id = class_to_id[current_class]
                cx = (x + icon.width / 2) / img_width
                cy = (y + icon.height / 2) / img_height
                w = icon.width / img_width
                h = icon.height / img_height
                annotations.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                break

        image_filename = f"{i:06d}.png"
        label_filename = f"{i:06d}.txt"
        background.save(os.path.join(images_out_dir, image_filename))
        with open(os.path.join(labels_out_dir, label_filename), 'w') as f:
            f.write("\n".join(annotations))

    print("Balanced dataset generation complete!")


if __name__ == '__main__':
    config = load_config()
    generate_dataset(config)
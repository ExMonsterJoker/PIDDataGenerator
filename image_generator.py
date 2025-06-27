import os
import json
import random
import itertools
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil
import yaml


def load_config(config_path='config/image_config.json'):
    """Loads the configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_icons_by_class(icons_dir):
    """Recursively finds all PNG icons and groups them by class."""
    icons_by_class = {}
    for root, _, files in os.walk(icons_dir):
        if not files: continue
        class_name = os.path.basename(root)
        if class_name not in icons_by_class:
            icons_by_class[class_name] = []
        for file in files:
            if file.lower().endswith('.png'):
                icons_by_class[class_name].append(os.path.join(root, file))
    return icons_by_class, sorted(icons_by_class.keys())


def organize_files_for_yolo(output_dir, all_images, split_config):
    """Splits files into train/val/test and returns a dict with the splits."""
    print("Organizing files into dataset splits...")
    random.shuffle(all_images)

    total_ratio = sum(split_config.values())
    if total_ratio == 0:
        print("Warning: All split ratios are 0. No files will be moved.")
        return {}
    split_ratios = {k: v / total_ratio for k, v in split_config.items()}

    num_total = len(all_images)
    num_train = int(num_total * split_ratios.get('train', 0))
    num_val = int(num_total * split_ratios.get('val', 0))

    train_files = all_images[:num_train]
    val_files = all_images[num_train:num_train + num_val]
    test_files = all_images[num_train + num_val:]

    splits = {'train': train_files, 'val': val_files}
    if test_files:
        splits['test'] = test_files

    temp_img_dir = os.path.join(output_dir, 'temp_images')
    temp_lbl_dir = os.path.join(output_dir, 'temp_labels')

    for split_name, file_list in splits.items():
        if not file_list: continue

        img_dir = os.path.join(output_dir, 'images', split_name)
        lbl_dir = os.path.join(output_dir, 'labels', split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for img_name in tqdm(file_list, desc=f"Moving {split_name} files"):
            shutil.move(os.path.join(temp_img_dir, img_name), img_dir)
            shutil.move(os.path.join(temp_lbl_dir, img_name.replace('.png', '.txt')), lbl_dir)

    shutil.rmtree(temp_img_dir)
    shutil.rmtree(temp_lbl_dir)

    return splits


def create_yaml_file(output_dir, class_names, included_splits):
    """Creates the data.yaml file based on the generated splits."""
    yaml_data = {'path': os.path.abspath(output_dir), 'train': 'images/train', 'val': 'images/val'}
    if 'test' in included_splits:
        yaml_data['test'] = 'images/test'
    yaml_data.update({'nc': len(class_names), 'names': class_names})

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
    print(f"Successfully created data.yaml at {yaml_path}")


def print_summary(class_counts, splits, output_dir):
    """Prints a final summary of the generated dataset."""
    print("\n" + "=" * 50)
    print("--- Dataset Generation Summary ---")
    print("=" * 50)

    print("\n[Object Class Distribution]")
    total_objects = 0
    for class_name, count in sorted(class_counts.items()):
        print(f"- {class_name:<20}: {count} instances")
        total_objects += count
    print("-----------------------------------")
    print(f"Total Objects Generated  : {total_objects}")

    print("\n[Image Splits]")
    total_images = 0
    for split_name, file_list in splits.items():
        count = len(file_list)
        print(f"- {split_name.capitalize():<20}: {count} images")
        total_images += count
    print("-----------------------------------")
    print(f"Total Images Generated   : {total_images}")

    print(f"\nDataset located at: {os.path.abspath(output_dir)}")
    print("=" * 50 + "\n")


def generate_dataset(config):
    """Generates the synthetic dataset based on the configuration."""
    output_dir = config['output']['dir']

    temp_img_dir = os.path.join(output_dir, 'temp_images')
    temp_lbl_dir = os.path.join(output_dir, 'temp_labels')
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(temp_img_dir)
    os.makedirs(temp_lbl_dir)

    icons_dir = config['input']['icons_dir']
    img_width, img_height = config['output']['image_size']
    num_images = config['output']['num_images']
    bg_color = tuple(config['input']['background_color'])
    min_objects = config['generation']['objects_per_image']['min']
    max_objects = config['generation']['objects_per_image']['max']
    allow_overlap = config['generation']['allow_overlap']
    aug_config = config.get('augmentation', {})

    icons_by_class, class_names = get_icons_by_class(icons_dir)
    if not class_names:
        print("Error: No icon classes found.");
        return

    class_to_id = {name: i for i, name in enumerate(class_names)}
    class_cycler = itertools.cycle(class_names)

    # --- Initialize counter for class balance summary ---
    class_counts = {name: 0 for name in class_names}
    generated_images = []

    for i in tqdm(range(num_images), desc="Generating images"):
        background = Image.new('RGB', (img_width, img_height), bg_color)
        num_objects = random.randint(min_objects, max_objects)
        placed_boxes = []
        annotations = []

        for _ in range(num_objects):
            current_class = next(class_cycler)
            # --- Increment class count for summary ---
            class_counts[current_class] += 1

            if not icons_by_class.get(current_class): continue
            icon_path = random.choice(icons_by_class[current_class])
            icon = Image.open(icon_path).convert("RGBA")

            if aug_config.get('enabled', False):
                scale_config = aug_config.get('scale', {})
                scale = random.uniform(scale_config.get('min', 1.0), scale_config.get('max', 1.0))
                if scale != 1.0: icon = icon.resize((int(icon.width * scale), int(icon.height * scale)),
                                                    Image.Resampling.LANCZOS)
                rot_config = aug_config.get('rotation', {})
                if rot_config.get('enabled', False) and random.random() < rot_config.get('chance', 0.0):
                    angle = random.choice(rot_config.get('angles', [0]))
                    if angle != 0: icon = icon.rotate(angle, expand=True, resample=Image.BICUBIC)
                hflip_config = aug_config.get('flip_horizontal', {})
                if hflip_config.get('enabled', False) and random.random() < hflip_config.get('chance',
                                                                                             0.0): icon = icon.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT)
                vflip_config = aug_config.get('flip_vertical', {})
                if vflip_config.get('enabled', False) and random.random() < vflip_config.get('chance',
                                                                                             0.0): icon = icon.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM)

            for _ in range(10):
                if icon.width > img_width or icon.height > img_height: break
                x = random.randint(0, img_width - icon.width)
                y = random.randint(0, img_height - icon.height)
                new_box = (x, y, x + icon.width, y + icon.height)

                is_overlapping = False
                if not allow_overlap:
                    for p_box in placed_boxes:
                        if not (new_box[0] > p_box[2] or new_box[2] < p_box[0] or new_box[1] > p_box[3] or new_box[3] <
                                p_box[1]):
                            is_overlapping = True;
                            break
                if is_overlapping: continue

                background.paste(icon, (x, y), icon)
                placed_boxes.append(new_box)
                class_id = class_to_id[current_class]
                cx, cy, w, h = (x + icon.width / 2) / img_width, (
                            y + icon.height / 2) / img_height, icon.width / img_width, icon.height / img_height
                annotations.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                break

        image_filename = f"pid_{i:06d}.png"
        label_filename = f"pid_{i:06d}.txt"
        background.save(os.path.join(temp_img_dir, image_filename))
        with open(os.path.join(temp_lbl_dir, label_filename), 'w') as f:
            f.write("\n".join(annotations))
        generated_images.append(image_filename)

    split_config = config.get('dataset_splits', {'train': 0.8, 'val': 0.2})
    final_splits = organize_files_for_yolo(output_dir, generated_images, split_config)
    create_yaml_file(output_dir, class_names, final_splits.keys())

    # --- Print the final summary ---
    print_summary(class_counts, final_splits, output_dir)
    print("\nDataset generation and organization complete!")


if __name__ == '__main__':
    # You will need the yaml dependency: pip install pyyaml
    config = load_config()
    generate_dataset(config)

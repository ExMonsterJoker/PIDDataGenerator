# visualizer.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def visualize_annotations(image_path, label_path, classes_file):
    """Visualize YOLO annotations on image"""

    # Load class names
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Load annotations
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    center_x = float(parts[1]) * width
                    center_y = float(parts[2]) * height
                    bbox_width = float(parts[3]) * width
                    bbox_height = float(parts[4]) * height

                    # Convert to corner coordinates
                    x1 = center_x - bbox_width / 2
                    y1 = center_y - bbox_height / 2

                    annotations.append({
                        'class_id': class_id,
                        'class_name': class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}',
                        'bbox': (x1, y1, bbox_width, bbox_height)
                    })

    # Plot
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

    for ann in annotations:
        x, y, w, h = ann['bbox']
        color = colors[ann['class_id'] % len(colors)]

        # Draw bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Add label
        ax.text(x, y - 5, ann['class_name'], fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(f"Annotations: {os.path.basename(image_path)}")
    plt.tight_layout()
    plt.show()


# data_validator.py
def validate_dataset(dataset_path):
    """Validate the generated dataset"""
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    classes_file = dataset_path / "classes.txt"

    print("Dataset Validation Report")
    print("=" * 50)

    # Check if directories exist
    if not images_dir.exists():
        print("❌ Images directory not found")
        return False

    if not labels_dir.exists():
        print("❌ Labels directory not found")
        return False

    if not classes_file.exists():
        print("❌ Classes file not found")
        return False

    # Count files
    image_files = list(images_dir.glob("*.jpg"))
    label_files = list(labels_dir.glob("*.txt"))

    print(f"📁 Images found: {len(image_files)}")
    print(f"📁 Labels found: {len(label_files)}")

    # Load classes
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"📁 Classes defined: {len(classes)}")

    # Validate annotations
    annotation_stats = {class_name: 0 for class_name in classes}
    invalid_annotations = 0

    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"⚠️  Invalid annotation format in {label_file.name}:{line_num}")
                    invalid_annotations += 1
                    continue

                try:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]

                    # Validate coordinates
                    if not all(0 <= coord <= 1 for coord in coords):
                        print(f"⚠️  Invalid coordinates in {label_file.name}:{line_num}")
                        invalid_annotations += 1
                        continue

                    # Count class occurrences
                    if 0 <= class_id < len(classes):
                        annotation_stats[classes[class_id]] += 1
                    else:
                        print(f"⚠️  Invalid class ID {class_id} in {label_file.name}:{line_num}")
                        invalid_annotations += 1

                except ValueError:
                    print(f"⚠️  Invalid number format in {label_file.name}:{line_num}")
                    invalid_annotations += 1

    # Print statistics
    print("\nClass Distribution:")
    print("-" * 30)
    total_annotations = sum(annotation_stats.values())
    for class_name, count in annotation_stats.items():
        percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
        print(f"{class_name}: {count} ({percentage:.1f}%)")

    print(f"\n📊 Total annotations: {total_annotations}")
    print(f"❌ Invalid annotations: {invalid_annotations}")

    # Check for orphaned files
    image_stems = {f.stem for f in image_files}
    label_stems = {f.stem for f in label_files}

    orphaned_images = image_stems - label_stems
    orphaned_labels = label_stems - image_stems

    if orphaned_images:
        print(f"⚠️  Images without labels: {len(orphaned_images)}")

    if orphaned_labels:
        print(f"⚠️  Labels without images: {len(orphaned_labels)}")

    print("\n" + "=" * 50)
    if invalid_annotations == 0 and not orphaned_images and not orphaned_labels:
        print("✅ Dataset validation passed!")
        return True
    else:
        print("❌ Dataset validation failed!")
        return False

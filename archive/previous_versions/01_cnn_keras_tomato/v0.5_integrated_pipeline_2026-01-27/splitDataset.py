import os
import shutil
import random

SOURCE_DIR = "/Users/kharulkamaaksha/Desktop/PlantVillage"
DEST_DIR   = "/Users/kharulkamaaksha/Desktop/PlantVillage_split"

TRAIN_RATIO = 0.8
random.seed(42)

train_dir = os.path.join(DEST_DIR, "train")
test_dir  = os.path.join(DEST_DIR, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    # Skip non-class folders
    if not os.path.isdir(class_path):
        continue
    if class_name.lower() in ["plantvillage", "train", "test"]:
        continue

    images = os.listdir(class_path)
    images = [img for img in images if os.path.isfile(os.path.join(class_path, img))]

    if len(images) == 0:
        continue

    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_idx]
    test_images  = images[split_idx:]

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_dir, class_name, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_dir, class_name, img)
        )

    print(f"{class_name}: {len(train_images)} train | {len(test_images)} test")

print("\nDataset split completed successfully!")

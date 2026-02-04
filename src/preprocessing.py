import os
import shutil
import random
import yaml


# ----------------------------
# Load parameters
# ----------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

SPLIT_RATIO = params["data"]["split_ratio"]  # Example: [0.7, 0.2, 0.1]

RAW_DIR = "data/raw"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"


# ----------------------------
# Create folder structure
# ----------------------------
def create_folders():
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(folder, exist_ok=True)


# ----------------------------
# Split dataset
# ----------------------------
def split_data():

    classes = os.listdir(RAW_DIR)

    for cls in classes:

        class_path = os.path.join(RAW_DIR, cls)
        images = os.listdir(class_path)

        random.shuffle(images)

        total = len(images)
        train_end = int(SPLIT_RATIO[0] * total)
        val_end = train_end + int(SPLIT_RATIO[1] * total)

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        for img_list, target_folder in [
            (train_imgs, TRAIN_DIR),
            (val_imgs, VAL_DIR),
            (test_imgs, TEST_DIR)
        ]:

            target_class_dir = os.path.join(target_folder, cls)
            os.makedirs(target_class_dir, exist_ok=True)

            for img in img_list:
                src = os.path.join(class_path, img)
                dst = os.path.join(target_class_dir, img)

                shutil.copy(src, dst)


# ----------------------------
# Run preprocessing
# ----------------------------
if __name__ == "__main__":

    print("Starting preprocessing...")

    create_folders()
    split_data()

    print("Data splitting completed.")

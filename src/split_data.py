import os
import shutil
import random

# -------- CONFIG --------
RAW_DIR = "data/raw"
OUTPUT_DIR = "data"

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(42)  # reproducibility
CLASSES = ["cancer", "normal"]

# -------- SPLITTING --------
for cls in CLASSES:
    src_path = os.path.join(RAW_DIR, cls)
    images = os.listdir(src_path)

    random.shuffle(images)

    total = len(images)
    train_end = int(total * SPLIT_RATIO["train"])
    val_end = train_end + int(total * SPLIT_RATIO["val"])

    split_data = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in split_data.items():
        dest_dir = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(dest_dir, exist_ok=True)

        for file in files:
            shutil.copy(
                os.path.join(src_path, file),
                os.path.join(dest_dir, file)
            )

print("✅ Dataset split complete: train / val / test created.")

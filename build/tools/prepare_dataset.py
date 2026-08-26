from pathlib import Path
import random
import shutil

ROOT = Path("dataset")

IMAGE_ROOT = ROOT / "images"
LABEL_ROOT = ROOT / "labels"

TRAIN_IMAGES = IMAGE_ROOT / "train"
VAL_IMAGES = IMAGE_ROOT / "val"

TRAIN_LABELS = LABEL_ROOT / "train"
VAL_LABELS = LABEL_ROOT / "val"

TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
VAL_IMAGES.mkdir(parents=True, exist_ok=True)
TRAIN_LABELS.mkdir(parents=True, exist_ok=True)
VAL_LABELS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Find all labeled images currently in dataset/images/
# ---------------------------------------------------------

pairs = []

for image in IMAGE_ROOT.glob("*.jpg"):

    label = LABEL_ROOT / f"{image.stem}.txt"

    if label.exists():
        pairs.append((image, label))

print(f"Labeled root images found: {len(pairs)}")

# ---------------------------------------------------------
# Shuffle
# ---------------------------------------------------------

random.seed(42)
random.shuffle(pairs)

# ---------------------------------------------------------
# 80/20 split
# ---------------------------------------------------------

split = int(len(pairs) * 0.8)

train_pairs = pairs[:split]
val_pairs = pairs[split:]

print(f"Train: {len(train_pairs)}")
print(f"Val:   {len(val_pairs)}")

# ---------------------------------------------------------
# Copy files
# ---------------------------------------------------------

for image, label in train_pairs:

    shutil.copy2(
        image,
        TRAIN_IMAGES / image.name
    )

    shutil.copy2(
        label,
        TRAIN_LABELS / label.name
    )

for image, label in val_pairs:

    shutil.copy2(
        image,
        VAL_IMAGES / image.name
    )

    shutil.copy2(
        label,
        VAL_LABELS / label.name
    )

print()
print("Dataset prepared successfully.")

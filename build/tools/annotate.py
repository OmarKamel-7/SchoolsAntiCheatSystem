from pathlib import Path
import cv2
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "runs/detect/train/weights/best.pt"

IMAGE_DIR = Path("dataset/images")
LABEL_DIR = Path("dataset/labels")

CONFIDENCE = 0.35

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded.")

# ============================================================
# FIND IMAGES THAT DO NOT HAVE LABELS
# ============================================================

images = sorted(IMAGE_DIR.glob("*.jpg"))

unlabeled = []

for image_path in images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        unlabeled.append(image_path)

print()
print(f"Total images:      {len(images)}")
print(f"Unlabeled images:  {len(unlabeled)}")
print()

if not unlabeled:
    print("Everything is already labeled.")
    exit()

# ============================================================
# PROCESS IMAGES
# ============================================================

for image_number, image_path in enumerate(unlabeled):

    print()
    print("=" * 60)
    print(
        f"Image {image_number + 1}/{len(unlabeled)}:"
        f" {image_path.name}"
    )
    print("=" * 60)

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image.")
        continue

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # YOLO prediction
    # --------------------------------------------------------

    results = model.predict(
        source=image,
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]

    boxes = result.boxes

    display = image.copy()

    best_box = None
    best_confidence = 0.0

    # --------------------------------------------------------
    # Find highest-confidence prediction
    # --------------------------------------------------------

    for box in boxes:

        confidence = float(box.conf[0])

        if confidence > best_confidence:

            best_confidence = confidence

            coordinates = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = coordinates

            best_box = (
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            )

    # --------------------------------------------------------
    # Draw prediction
    # --------------------------------------------------------

    if best_box is not None:

        x1, y1, x2, y2 = best_box

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        text = f"AirPod {best_confidence:.2f}"

        cv2.putText(
            display,
            text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            display,
            "NO DETECTION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    cv2.putText(
        display,
        "A = accept    R = redraw    S = skip    Q = quit",
        (10, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Show image
    # --------------------------------------------------------

    cv2.imshow(
        "AirPod Annotation",
        display
    )

    key = cv2.waitKey(0) & 0xFF

    # ========================================================
    # ACCEPT PREDICTION
    # ========================================================

    if key == ord("a") or key == ord("A"):

        if best_box is None:

            print("No prediction to accept.")

            continue

        x1, y1, x2, y2 = best_box

        # Convert to YOLO normalized format

        center_x = ((x1 + x2) / 2) / width
        center_y = ((y1 + y2) / 2) / height

        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height

        label_path = (
            LABEL_DIR /
            f"{image_path.stem}.txt"
        )

        with open(label_path, "w") as f:

            f.write(
                f"0 "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}\n"
            )

        print(
            f"Saved label: {label_path}"
        )

    # ========================================================
    # REDRAW BOX
    # ========================================================

    elif key == ord("r") or key == ord("R"):

        print("Draw a box around the AirPod.")

        roi = cv2.selectROI(
            "Draw AirPod",
            image,
            fromCenter=False,
            showCrosshair=True
        )

        cv2.destroyWindow("Draw AirPod")

        x, y, w, h = roi

        if w <= 0 or h <= 0:

            print("Invalid box. Skipping.")

            continue

        # Convert to YOLO format

        center_x = (x + w / 2) / width
        center_y = (y + h / 2) / height

        box_width = w / width
        box_height = h / height

        label_path = (
            LABEL_DIR /
            f"{image_path.stem}.txt"
        )

        with open(label_path, "w") as f:

            f.write(
                f"0 "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}\n"
            )

        print(
            f"Saved manual label: {label_path}"
        )

    # ========================================================
    # SKIP
    # ========================================================

    elif key == ord("s") or key == ord("S"):

        print("Skipped.")

    # ========================================================
    # QUIT
    # ========================================================

    elif key == ord("q") or key == ord("Q"):

        print("Stopping annotation.")

        break

    cv2.destroyAllWindows()

cv2.destroyAllWindows()

print()
print("Annotation session finished.")

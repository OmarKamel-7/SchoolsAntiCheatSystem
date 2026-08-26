import cv2
import os
import time

SAVE_DIR = "dataset/new_images"

os.makedirs(SAVE_DIR, exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit(1)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0

# Continue numbering after existing images
existing = [
    f for f in os.listdir(SAVE_DIR)
    if f.endswith(".jpg")
]

count = len(existing) + 1

print()
print("======================================")
print("      AIRPOD IMAGE CAPTURE")
print("======================================")
print()
print("SPACE = take photo")
print("Q     = quit")
print()

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    text = f"Photos: {count - 1}"

    cv2.putText(
        frame,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SPACE = capture | Q = quit",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Capture AirPod Images", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break

    if key == 32:  # SPACE

        filename = f"airpod_new_{count:04d}.jpg"

        path = os.path.join(
            SAVE_DIR,
            filename
        )

        cv2.imwrite(path, frame)

        print(f"Saved: {path}")

        count += 1

        # Small delay prevents accidental double captures
        time.sleep(0.2)

camera.release()
cv2.destroyAllWindows()

print()
print("Capture finished.")

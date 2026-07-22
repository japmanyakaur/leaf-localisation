"""
Live webcam demo: point a camera at a leaf and see the detector draw a box
around it in real time, with area (in pixels) printed on screen.

Press 'q' to quit.

Usage:
    python scripts/webcam_demo.py --weights runs/detect/train/weights/best.pt
"""
import argparse

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/train/weights/best.pt",
                         help="path to trained model weights (best.pt)")
    parser.add_argument("--camera", type=int, default=1,
                         help="camera index ")
    args = parser.parse_args()

    model = YOLO(args.weights)
    cam = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        annotated = results[0].plot()

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area_px = (x2 - x1) * (y2 - y1)
            cv2.putText(annotated, f"{area_px:.0f}px^2", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Leaf Localisation", annotated)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

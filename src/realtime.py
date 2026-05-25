from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.pipeline import CombinedRoadSignRecognizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 7: Real-time webcam detection")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--detection-conf", type=float, default=0.4)
    parser.add_argument("--yolo-model", type=Path, default=PROJECT_ROOT / "models" / "yolo_best.pt")
    parser.add_argument("--cnn-model", type=Path, default=PROJECT_ROOT / "models" / "best_cnn_model.pth")
    parser.add_argument("--class-names", type=Path, default=PROJECT_ROOT / "artifacts" / "class_names.json")
    parser.add_argument("--use-imagenet-norm", action="store_true")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    recognizer = CombinedRoadSignRecognizer(
        yolo_weights=args.yolo_model,
        cnn_model_path=args.cnn_model,
        class_names_path=args.class_names if args.class_names.exists() else None,
        use_imagenet_norm=True if args.use_imagenet_norm else None,
        device=args.device,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam index {args.camera}")

    print("Press 'q' to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result_frame, _ = recognizer.predict_frame(frame, detection_conf=args.detection_conf)
        cv2.imshow("Road Sign Detection", result_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
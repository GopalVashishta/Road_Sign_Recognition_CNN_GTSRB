from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.cnn_model import (
    build_transforms,
    load_classifier_checkpoint,
    resolve_device,
    tensor_from_bgr_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "models" / "yolo_best.pt"
DEFAULT_CNN_MODEL = PROJECT_ROOT / "models" / "best_cnn_model.pth"
DEFAULT_CLASS_NAMES = PROJECT_ROOT / "artifacts" / "class_names.json"
DEFAULT_CNN_LABEL_MAP = PROJECT_ROOT / "artifacts" / "cnn_label_map.json"
COMMON_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5: Combined YOLO + CNN inference")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "prediction.png")
    parser.add_argument("--detection-conf", type=float, default=0.4)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--cnn-model", type=Path, default=DEFAULT_CNN_MODEL)
    parser.add_argument("--class-names", type=Path, default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--use-imagenet-norm", action="store_true")
    parser.add_argument(
        "--force-cnn-on-mismatch",
        action="store_true",
        help="Deprecated: pipeline now always uses CNN labels for detected boxes.",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_class_names(
    class_names_path: Optional[Path],
    num_classes: int,
    fallback_names: Optional[List[str]] = None,
) -> List[str]:
    if class_names_path is not None and class_names_path.exists():
        try:
            names = json.loads(class_names_path.read_text(encoding="utf-8"))
            if isinstance(names, list) and len(names) == num_classes:
                return [str(x) for x in names]
        except json.JSONDecodeError:
            pass

    if fallback_names is not None and len(fallback_names) == num_classes:
        return [str(x) for x in fallback_names]

    return [f"class_{idx}" for idx in range(num_classes)]


def load_imagenet_norm_flag(default_value: bool = False) -> bool:
    if not DEFAULT_CNN_LABEL_MAP.exists():
        return default_value
    try:
        payload = json.loads(DEFAULT_CNN_LABEL_MAP.read_text(encoding="utf-8"))
        if "uses_imagenet_normalization" in payload:
            return bool(payload.get("uses_imagenet_normalization", default_value))
        return bool(payload.get("uses_mobilenet_preprocess", default_value))
    except json.JSONDecodeError:
        return default_value


def _normalize_label_name(name: str) -> str:
    lowered = name.strip().lower()
    return lowered.replace("-", "_").replace(" ", "_")


def _extract_detector_names(detector: YOLO) -> List[str]:
    raw_names = getattr(detector, "names", None)
    if isinstance(raw_names, dict):
        try:
            return [str(raw_names[idx]) for idx in sorted(raw_names, key=int)]
        except Exception:
            return [str(value) for _, value in sorted(raw_names.items(), key=lambda kv: str(kv[0]))]
    if isinstance(raw_names, list):
        return [str(item) for item in raw_names]
    return []


def resolve_image_path(image_path: Path) -> Path:
    attempted: List[Path] = []

    direct_candidates = [image_path]
    if not image_path.is_absolute():
        direct_candidates.append((PROJECT_ROOT / image_path).resolve())

    for candidate in direct_candidates:
        attempted.append(candidate)
        if candidate.exists():
            return candidate

        stem = candidate.stem
        parent = candidate.parent
        for ext in COMMON_IMAGE_EXTENSIONS:
            alt = parent / f"{stem}{ext}"
            attempted.append(alt)
            if alt.exists():
                return alt

    stem = image_path.stem
    fallback_dirs = [
        PROJECT_ROOT / "test",
        PROJECT_ROOT / "dataset" / "Test",
        PROJECT_ROOT / "yolo_dataset" / "images" / "val",
        PROJECT_ROOT / "yolo_dataset" / "images" / "train",
    ]

    seen = set()
    for folder in fallback_dirs:
        folder_key = str(folder.resolve()).lower()
        if folder_key in seen:
            continue
        seen.add(folder_key)

        for ext in COMMON_IMAGE_EXTENSIONS:
            alt = folder / f"{stem}{ext}"
            attempted.append(alt)
            if alt.exists():
                return alt

    attempted_preview = ", ".join(str(path) for path in attempted[:12])
    raise FileNotFoundError(
        f"Input image not found: {image_path}. Tried: {attempted_preview}"
    )


class CombinedRoadSignRecognizer:
    def __init__(
        self,
        yolo_weights: Path,
        cnn_model_path: Path,
        class_names_path: Optional[Path] = None,
        use_imagenet_norm: Optional[bool] = None,
        force_cnn_on_mismatch: bool = False,
        device: str = "",
    ) -> None:
        if not yolo_weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")
        if not cnn_model_path.exists():
            raise FileNotFoundError(f"CNN model not found: {cnn_model_path}")

        self.device = resolve_device(device)
        self.detector = YOLO(str(yolo_weights))
        self.detector_names = _extract_detector_names(self.detector)

        self.classifier, metadata = load_classifier_checkpoint(cnn_model_path, device=self.device)
        self.image_size = metadata.image_size

        if use_imagenet_norm is None:
            self.use_imagenet_norm = metadata.use_imagenet_normalization
        else:
            self.use_imagenet_norm = use_imagenet_norm

        self.classifier_transform = build_transforms(
            image_size=self.image_size,
            use_imagenet_normalization=self.use_imagenet_norm,
            train=False,
        )

        if class_names_path is None:
            class_names_path = DEFAULT_CLASS_NAMES

        self.class_names = load_class_names(
            class_names_path=class_names_path,
            num_classes=metadata.num_classes,
            fallback_names=metadata.class_names,
        )

        self.detector_num_classes = len(self.detector_names)
        self.cnn_num_classes = int(metadata.num_classes)

        if self.detector_num_classes == 0:
            self.class_spaces_match = True
        elif self.detector_num_classes != self.cnn_num_classes:
            self.class_spaces_match = False
        else:
            detector_norm = [_normalize_label_name(name) for name in self.detector_names]
            cnn_norm = [_normalize_label_name(name) for name in self.class_names]
            self.class_spaces_match = detector_norm == cnn_norm

        self.force_cnn_on_mismatch = force_cnn_on_mismatch
        # Pipeline mode: YOLO proposes boxes, CNN always performs final classification.
        self.use_cnn_labels = True

    def _prepare_classifier_input(self, crop_bgr: np.ndarray) -> torch.Tensor:
        return tensor_from_bgr_image(crop_bgr, self.classifier_transform).to(self.device)

    def classify_crop(self, crop_bgr: np.ndarray) -> Tuple[int, float]:
        model_input = self._prepare_classifier_input(crop_bgr)
        with torch.no_grad():
            logits = self.classifier(model_input)
            probabilities = torch.softmax(logits, dim=1)[0]
        class_idx = int(torch.argmax(probabilities).item())
        class_conf = float(probabilities[class_idx].item())
        return class_idx, class_conf

    def predict_frame(self, frame_bgr: np.ndarray, detection_conf: float = 0.4) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        output_frame = frame_bgr.copy()
        detections: List[Dict[str, object]] = []

        results = self.detector(frame_bgr, verbose=False)
        if not results:
            return output_frame, detections

        h, w = output_frame.shape[:2]
        boxes = results[0].boxes
        for box in boxes:
            det_conf = float(box.conf[0])
            if det_conf < detection_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            det_class_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else -1
            if 0 <= det_class_id < len(self.detector_names):
                det_class_name = self.detector_names[det_class_id]
            else:
                det_class_name = f"class_{det_class_id}" if det_class_id >= 0 else "detected_sign"

            # OLD behavior (detector-label fallback on class-space mismatch):
            # class_idx = det_class_id
            # class_name = det_class_name
            # class_conf = det_conf
            # shown_conf = det_conf
            # label_source = "detector"
            # New behavior: always classify each detected box with the CNN.
            class_idx, class_conf = self.classify_crop(crop)
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"class_{class_idx}"
            shown_conf = class_conf
            label_source = "cnn"

            label = f"{class_name} ({shown_conf:.2f})"
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output_frame,
                label,
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "label_source": label_source,
                    "detector_class_id": det_class_id,
                    "detector_class_name": det_class_name,
                    "detector_confidence": round(det_conf, 4),
                    "class_id": class_idx,
                    "class_name": class_name,
                    "classifier_confidence": round(class_conf, 4),
                    "fused_confidence": round(shown_conf, 4),
                }
            )

        return output_frame, detections

    def predict_image(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        detection_conf: float = 0.4,
    ) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        resolved_image_path = resolve_image_path(image_path)
        if resolved_image_path != image_path:
            print(f"[Info] Input image not found at '{image_path}', using '{resolved_image_path}'")

        image = cv2.imread(str(resolved_image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {resolved_image_path}")

        result, detections = self.predict_frame(image, detection_conf=detection_conf)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), result)
        return result, detections


def main() -> None:
    args = parse_args()

    recognizer = CombinedRoadSignRecognizer(
        yolo_weights=args.yolo_model,
        cnn_model_path=args.cnn_model,
        class_names_path=args.class_names if args.class_names.exists() else None,
        use_imagenet_norm=True if args.use_imagenet_norm else None,
        force_cnn_on_mismatch=args.force_cnn_on_mismatch,
        device=args.device,
    )

    print("Pipeline mode: YOLO boxes -> CNN classification")
    if not recognizer.class_spaces_match:
        print(
            f"[Info] Class spaces differ (YOLO={recognizer.detector_num_classes}, "
            f"CNN={recognizer.cnn_num_classes}), but CNN labels are still used."
        )

    result, detections = recognizer.predict_image(
        image_path=args.image,
        output_path=args.output,
        detection_conf=args.detection_conf,
    )

    print(f"Saved prediction image to: {args.output}")
    print(f"Detections: {len(detections)}")
    for item in detections:
        print(item)

    if args.show:
        cv2.imshow("Road Sign Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
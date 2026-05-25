from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.pipeline import CombinedRoadSignRecognizer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "models" / "yolo_best.pt"
DEFAULT_CNN_MODEL = PROJECT_ROOT / "models" / "best_cnn_model.pth"
DEFAULT_CLASS_NAMES = PROJECT_ROOT / "artifacts" / "class_names.json"


@st.cache_resource
def load_pipeline(
    yolo_model: Path,
    cnn_model: Path,
    class_names: Path,
    use_imagenet_norm: Optional[bool],
) -> CombinedRoadSignRecognizer:
    return CombinedRoadSignRecognizer(
        yolo_weights=yolo_model,
        cnn_model_path=cnn_model,
        class_names_path=class_names if class_names.exists() else None,
        use_imagenet_norm=use_imagenet_norm,
    )


def main() -> None:
    st.set_page_config(page_title="Road Sign Recognition", layout="wide")
    st.title("Road Sign Recognition: YOLOv8 + CNN")

    st.sidebar.header("Settings")
    detection_conf = st.sidebar.slider("Detection Confidence", 0.1, 0.95, 0.4, 0.05)
    force_imagenet_norm = st.sidebar.checkbox("Force ImageNet normalization", value=False)

    if not DEFAULT_YOLO_MODEL.exists() or not DEFAULT_CNN_MODEL.exists():
        st.warning(
            "Model files are missing. Train models first:\n"
            "- python -m src.train_cnn\n"
            "- python -m src.train_yolo"
        )
        return

    recognizer = load_pipeline(
        yolo_model=DEFAULT_YOLO_MODEL,
        cnn_model=DEFAULT_CNN_MODEL,
        class_names=DEFAULT_CLASS_NAMES,
        use_imagenet_norm=True if force_imagenet_norm else None,
    )

    uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload an image to run detection + classification.")
        return

    pil_image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    result_bgr, detections = recognizer.predict_frame(image_bgr, detection_conf=detection_conf)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(result_rgb, caption="Detected road signs", use_container_width=True)

    with col2:
        st.subheader("Detections")
        if not detections:
            st.write("No signs detected.")
        else:
            st.dataframe(detections, use_container_width=True)


if __name__ == "__main__":
    main()
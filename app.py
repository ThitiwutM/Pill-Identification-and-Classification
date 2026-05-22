import streamlit as st
import numpy as np
import joblib

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

from utils import (
    extract_color_features,
    extract_lbp_features,
    extract_shape_features
)

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Pill Classification",
    layout="wide"
)

# =====================================
# DOWNLOAD MODELS
# =====================================

YOLO_PATH = hf_hub_download(
    repo_id="ZunTM/pill-classification-hybrid",
    filename="best.pt"
)

RF_PATH = hf_hub_download(
    repo_id="ZunTM/pill-classification-hybrid",
    filename="random_forest.pkl"
)

# =====================================
# LOAD MODELS
# =====================================

yolo_model = YOLO(YOLO_PATH)

rf_model = joblib.load(RF_PATH)

# =====================================
# CLASS NAMES
# =====================================

class_names = [
    'Neozep',
    'Biogesic',
    'Fishoil',
    'Medicol',
    'Bactidol',
    'Bioflu',
    'Kremil S',
    'Alaxan',
    'Decolgen',
    'DayZinc'
]

# =====================================
# TITLE
# =====================================

st.title(
    "Pill Identification and Classification"
)

st.markdown(
    """
    Hybrid AI Framework using:
    - YOLOv8s
    - Random Forest
    - Feature Fusion
    """
)

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Upload Pill Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# MAIN PIPELINE
# =====================================

if uploaded_file is not None:

    # =========================
    # LOAD IMAGE
    # =========================

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    img_rgb = np.array(image)

    h_img, w_img = img_rgb.shape[:2]

    # =========================
    # YOLO DETECTION
    # =========================

    results = yolo_model.predict(
        source=img_rgb,
        conf=0.10,
        imgsz=1280,
        augment=True,
        verbose=False
    )

    # =========================
    # DRAW OBJECT
    # =========================

    draw = ImageDraw.Draw(image)

    # =========================
    # LOOP DETECTIONS
    # =========================

    for r in results:

        pred_boxes = r.boxes.xyxy.cpu().numpy()

        pred_classes = r.boxes.cls.cpu().numpy()

        pred_conf = r.boxes.conf.cpu().numpy()

        for pred_box, yolo_cls, conf in zip(
            pred_boxes,
            pred_classes,
            pred_conf
        ):

            pred_box = list(
                map(int, pred_box)
            )

            x1, y1, x2, y2 = pred_box

            # =========================
            # CLAMP
            # =========================

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w_img, x2)
            y2 = min(h_img, y2)

            # =========================
            # CROP ROI
            # =========================

            crop = img_rgb[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            if (
                crop.shape[0] < 8
                or crop.shape[1] < 8
            ):
                continue

            # =========================
            # RESIZE
            # =========================

            crop_pil = Image.fromarray(
                crop
            )

            crop_pil = crop_pil.resize(
                (128,128)
            )

            crop = np.array(crop_pil)

            # =========================
            # FEATURE EXTRACTION
            # =========================

            color_feature = (
                extract_color_features(crop)
            )

            texture_feature = (
                extract_lbp_features(crop)
            )

            shape_feature = (
                extract_shape_features(crop)
            )

            final_feature = np.concatenate([
                color_feature,
                texture_feature,
                shape_feature
            ])

            final_feature = (
                final_feature.reshape(1,-1)
            )

            # =========================
            # RF PREDICTION
            # =========================

            rf_probs = (
                rf_model.predict_proba(
                    final_feature
                )[0]
            )

            rf_pred = np.argmax(
                rf_probs
            )

            rf_conf = np.max(
                rf_probs
            )

            # =========================
            # FUSION LOGIC
            # =========================

            if conf >= 0.95:

                final_pred = int(
                    yolo_cls
                )

            elif (
                conf >= 0.60
                and int(yolo_cls)
                == int(rf_pred)
                and rf_conf >= 0.60
            ):

                final_pred = int(
                    yolo_cls
                )

            else:

                continue

            final_name = class_names[
                int(final_pred)
            ]

            # =========================
            # DRAW BOX
            # =========================

            draw.rectangle(
                [
                    (x1,y1),
                    (x2,y2)
                ],
                outline="lime",
                width=4
            )

            draw.text(
                (x1, y1-20),
                final_name,
                fill="red"
            )

    # =========================
    # SHOW IMAGE
    # =========================

    st.image(
        image,
        caption="Prediction Result",
        use_container_width=True
    )

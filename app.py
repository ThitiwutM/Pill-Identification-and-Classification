import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import joblib
from PIL import Image
from utils import (
    extract_color_features,
    extract_lbp_features,
    extract_shape_features
)

# =========================
# LOAD MODELS
# =========================

yolo_model = YOLO('best.pt')

rf_model = joblib.load('random_forest.pkl')

# =========================
# CLASS NAMES
# =========================

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

# =========================
# STREAMLIT UI
# =========================

st.title("Pill Identification and Classification")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    img = np.array(image)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h_img, w_img = img_rgb.shape[:2]

    # =========================
    # YOLO DETECTION
    # =========================

    results = yolo_model.predict(
        source=img_rgb,
        conf=0.1,
        imgsz=640,
        verbose=False
    )

    for r in results:

        pred_boxes = r.boxes.xyxy.cpu().numpy()

        pred_classes = r.boxes.cls.cpu().numpy()

        pred_conf = r.boxes.conf.cpu().numpy()

        for pred_box, yolo_cls, conf in zip(
            pred_boxes,
            pred_classes,
            pred_conf
        ):

            pred_box = list(map(int, pred_box))

            x1, y1, x2, y2 = pred_box

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w_img, x2)
            y2 = min(h_img, y2)

            crop = img_rgb[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            crop = cv2.resize(crop, (128,128))

            # =========================
            # FEATURE EXTRACTION
            # =========================

            color_feature = extract_color_features(crop)

            texture_feature = extract_lbp_features(crop)

            shape_feature = extract_shape_features(crop)

            final_feature = np.concatenate([
                color_feature,
                texture_feature,
                shape_feature
            ])

            final_feature = final_feature.reshape(1, -1)

            # =========================
            # RF PREDICTION
            # =========================

            rf_probs = rf_model.predict_proba(
                final_feature
            )[0]

            rf_pred = np.argmax(rf_probs)

            rf_conf = np.max(rf_probs)

            # =========================
            # FUSION LOGIC
            # =========================

            if conf >= 0.95:

                final_pred = int(yolo_cls)

            elif (
                conf >= 0.60
                and int(yolo_cls) == int(rf_pred)
                and rf_conf >= 0.60
            ):

                final_pred = int(yolo_cls)

            else:

                continue

            final_name = class_names[int(final_pred)]

            # =========================
            # DRAW BOX
            # =========================

            cv2.rectangle(
                img_rgb,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                3
            )

            cv2.putText(
                img_rgb,
                final_name,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,0,0),
                2
            )

    st.image(
        img_rgb,
        caption='Prediction Result',
        use_container_width=True
    )

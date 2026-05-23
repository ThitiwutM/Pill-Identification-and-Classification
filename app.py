import streamlit as st
import numpy as np

from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from PIL import Image, ImageDraw

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Pill Detection",
    layout="wide"
)

# =====================================
# DOWNLOAD MODEL
# =====================================

MODEL_PATH = hf_hub_download(
    repo_id="ZunTM/pill-classification-hybrid",
    filename="best.pt"
)

# =====================================
# LOAD MODEL
# =====================================

model = YOLO(MODEL_PATH)

# =====================================
# TITLE
# =====================================

st.title(
    "Pill Identification using YOLOv8s"
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# MAIN PIPELINE
# =====================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    img = np.array(image)

    # =====================================
    # YOLO PREDICTION
    # =====================================

    results = model.predict(
        source=img,
        conf=0.25,
        imgsz=1280,
        augment=True,
        verbose=False
    )

    # =====================================
    # DRAW BOXES MANUALLY
    # =====================================

    draw = ImageDraw.Draw(image)

    for r in results:

        boxes = r.boxes.xyxy.cpu().numpy()

        classes = r.boxes.cls.cpu().numpy()

        confs = r.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(
            boxes,
            classes,
            confs
        ):

            x1, y1, x2, y2 = map(int, box)

            label = (
                f"{model.names[int(cls)]}"
            )

            # DRAW BOX

            draw.rectangle(
                [(x1,y1),(x2,y2)],
                outline="lime",
                width=4
            )

            # DRAW TEXT

            draw.text(
                (x1, y1-20),
                label,
                fill="red"
            )

    # =====================================
    # SHOW ORIGINAL IMAGE
    # =====================================

    st.image(
        image,
        caption="Detection Result",
        use_container_width=True
    )

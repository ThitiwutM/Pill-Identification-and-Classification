import streamlit as st
import numpy as np

from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from PIL import Image

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

st.markdown(
    """
    Upload an image for pill detection and classification
    """
)

# =====================================
# FILE UPLOADER
# =====================================

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
        conf=0.90,
        imgsz=1280,
        augment=True,
        verbose=False
    )

    # =====================================
    # DRAW RESULTS
    # =====================================

    plotted = results[0].plot()

    plotted = plotted[..., ::-1]

    # =====================================
    # SHOW IMAGE
    # =====================================

    st.image(
        plotted,
        caption="Detection Result",
        use_container_width=True
    )

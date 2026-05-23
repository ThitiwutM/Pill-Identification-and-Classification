import streamlit as st
import numpy as np
import tempfile

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

    # =====================================
    # READ IMAGE
    # =====================================

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # =====================================
    # SAVE TEMP FILE
    # =====================================

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as tmp:

        image.save(tmp.name)

        temp_path = tmp.name

    # =====================================
    # YOLO PREDICTION
    # =====================================

    results = model.predict(
        source=temp_path,
        conf=0.9,
        iou=0.45,
        verbose=False
    )

    # =====================================
    # DRAW RESULTS
    # =====================================

    plotted = results[0].plot(
        conf=True,
        line_width=2,
        font_size=10
    )

    plotted = plotted[:, :, ::-1]

    # =====================================
    # SHOW RESULT
    # =====================================

    st.image(
        plotted,
        caption="Detection Result",
        use_container_width=True
    )

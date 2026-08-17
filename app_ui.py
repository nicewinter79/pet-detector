"""
app_ui.py — Streamlit dashboard for the pet-detector YOLO model.
Run with: streamlit run app_ui.py
"""

import tempfile
from collections import Counter
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "../runs/detect/train/weights/best.pt"


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def run_detection(model, image_path: str, conf: float):
    results = model.predict(source=image_path, conf=conf, save=False)
    result = results[0]

    counts = Counter()
    for cls_idx in result.boxes.cls:
        class_name = model.names[int(cls_idx)]
        counts[class_name] += 1

    return result, counts


def main():
    st.set_page_config(page_title="Pet Detector", page_icon="🐾", layout="centered")
    st.title("🐾 Pet Detector")
    st.write("Upload a photo and the model will detect and count the animals in it.")

    if not Path(MODEL_PATH).exists():
        st.error(f"Model weights not found at `{MODEL_PATH}`.")
        return

    model = load_model(MODEL_PATH)
    conf = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Running detection..."):
            result, counts = run_detection(model, tmp_path, conf)

        annotated = result.plot()[:, :, ::-1]
        st.image(annotated, caption="Detections", use_container_width=True)

        st.subheader("Counts")
        if not counts:
            st.write("No animals detected. Try lowering the confidence threshold.")
        else:
            total = sum(counts.values())
            cols = st.columns(len(counts))
            for col, (class_name, count) in zip(cols, sorted(counts.items(), key=lambda x: -x[1])):
                col.metric(class_name, count)
            st.write(f"**Total animals detected:** {total}")


if __name__ == "__main__":
    main()
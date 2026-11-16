import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

DATA_DIR = Path("skin_cancer_dataset")
MODEL_PATH = Path("skin_cancer_cnn.keras")
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32

st.set_page_config(page_title="Skin Image Classifier", page_icon="🔬")
st.title("Skin Image Classifier — Educational Project")
st.warning(
    "Educational demonstration only. This app is NOT a medical diagnostic tool. "
    "Its output must not be used to decide whether a skin mark is cancerous. "
    "For any concerning or changing skin mark, consult a qualified healthcare professional."
)

def get_class_names():
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and not p.name.startswith("."))

def train_model():
    class_names = get_class_names()
    if len(class_names) < 2:
        st.error(
            "Dataset not ready. Add at least two category folders inside "
            "skin_cancer_dataset, with labeled JPG/PNG images in each folder."
        )
        return False

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=IMAGE_SIZE + (3,)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=8)
    model.save(MODEL_PATH)
    return True

class_names = get_class_names()
st.subheader("1. Prepare and train")
st.write("Dataset categories found:", ", ".join(class_names) if class_names else "None yet")
if st.button("Train / Create CNN model", type="primary"):
    with st.spinner("Training model. This may take a while..."):
        if train_model():
            st.success("Model created: skin_cancer_cnn.keras")
            st.rerun()

st.subheader("2. Upload an image for model output")
if not MODEL_PATH.exists():
    st.info("First add your real labeled dataset and train the model above.")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
uploaded = st.file_uploader("Upload JPG or PNG image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Run model"):
        resized = image.resize(IMAGE_SIZE)
        arr = np.asarray(resized, dtype=np.float32)[None, ...]
        scores = model.predict(arr, verbose=0)[0]
        names = get_class_names()
        if len(names) != len(scores):
            st.error("Dataset folders no longer match this model. Retrain the model.")
        else:
            idx = int(np.argmax(scores))
            st.write("Dataset category predicted:", names[idx])
            st.write(f"Model score: {scores[idx] * 100:.1f}%")
            st.caption("A model score is not a medical diagnosis or cancer probability.")

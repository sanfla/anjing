import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

st.title("Klasifikasi Gambar: Kucing vs Anjing")

model_path = hf_hub_download(
    repo_id="sanfla/anjingKucing", 
    filename="cnnModel.h5"
)

model = tf.keras.models.load_model(model_path, compile=False)
st.write("Input shape model:", model.input_shape)

def preprocess(img):
    target_size = model.input_shape[1:3]
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

uploaded = st.file_uploader("Upload gambar", type=['jpg', 'png', 'jpeg'])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar diunggah", use_column_width=True)
    input_img = preprocess(img)

    pred = model.predict(input_img)

    if pred.shape[1] == 1:  
        label = "Kucing" if pred[0][0] > 0.5 else "Anjing"
    else:  
        label = "Kucing" if np.argmax(pred[0]) == 0 else "Anjing"

    st.subheader(f"Prediksi: {label}")

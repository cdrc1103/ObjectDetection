import requests
import streamlit as st
from PIL import Image

st.title("Object Detection")


image_upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
model_option = st.selectbox("Choose a model", ["yolov3tiny", "yolov3"])

if st.button("Detect Objects") and image_upload is not None:
    files = {"file": image_upload.getvalue()}
    response = requests.post(
        f"http://127.0.0.1:8000/predict/?model={model_option}", files=files
    )
    image_path = response.json()
    image = Image.open(image_path.get("name"))
    st.image(image, width=500)

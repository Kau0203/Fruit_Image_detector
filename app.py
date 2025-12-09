import streamlit as st
from PIL import Image
import torch

# Load your model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# ----------------- BACKGROUND IMAGE -----------------
def set_background(image_url):
    page_bg = f"""
    <style>
    .stApp {{
        background: url("https://img.freepik.com/free-vector/hand-drawn-fruits-vegetables-pattern-background_23-2150855720.jpg?semt=ais_hybrid&w=740&q=80");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# ---------- ADD YOUR BACKGROUND IMAGE URL HERE ----------
BACKGROUND_IMAGE_URL = "https://your-image-link.com/image.jpg"
set_background(BACKGROUND_IMAGE_URL)
# ---------------------------------------------------------

st.title("Fruit Image Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run model
    results = model(image)
    results.render()

    st.image(results.ims[0], caption="Prediction", use_column_width=True)

    # Show labels
    st.write("Detected:", results.pandas().xyxy[0])

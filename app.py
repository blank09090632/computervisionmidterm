import cv2
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Processing App", layout="centered")
st.title("Image Processing App")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = np.array(Image.open(uploaded))
    st.image(img, caption="Original Image", use_column_width=True)

    op = st.selectbox(
        "Choose image operation:",
        [
            "Grayscale",
            "Blur",
            "Sharpen",
            "Edge Detection (Canny)",
            "Brightness Adjust",
            "Contrast Adjust",
            "Rotate",
            "Flip (Horizontal)",
            "Flip (Vertical)",
            "Denoise - Median Filter",
            "Denoise - Gaussian",
            "Denoise - Bilateral Filter",
            "Resize Image",
        ]
    )

    processed = None

    
    if op == "Grayscale":
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif op == "Blur":
        processed = cv2.GaussianBlur(img, (15, 15), 0)

    elif op == "Sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        processed = cv2.filter2D(img, -1, kernel)

    elif op == "Edge Detection (Canny)":
        processed = cv2.Canny(img, 100, 200)

    elif op == "Brightness Adjust":
        v = st.slider("Brightness", -100, 100, 0)
        processed = cv2.convertScaleAbs(img, alpha=1, beta=v)

    elif op == "Contrast Adjust":
        v = st.slider("Contrast", 0.5, 3.0, 1.0)
        processed = cv2.convertScaleAbs(img, alpha=v, beta=0)

    elif op == "Rotate":
        angle = st.slider("Rotation Angle", -180, 180, 0)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        processed = cv2.warpAffine(img, M, (w, h))

    elif op == "Flip (Horizontal)":
        processed = cv2.flip(img, 1)

    elif op == "Flip (Vertical)":
        processed = cv2.flip(img, 0)

    
    elif op == "Denoise - Median Filter":
        k = st.slider("Kernel Size (odd numbers only)", 1, 15, 5, step=2)
        processed = cv2.medianBlur(img, k)

    elif op == "Denoise - Gaussian":
        k = st.slider("Kernel Size (odd numbers only)", 1, 15, 5, step=2)
        processed = cv2.GaussianBlur(img, (k, k), 0)

    elif op == "Denoise - Bilateral Filter":
        d = st.slider("Diameter", 1, 15, 5)
        sc = st.slider("Sigma Color", 10, 150, 75)
        ss = st.slider("Sigma Space", 10, 150, 75)
        processed = cv2.bilateralFilter(img, d, sc, ss)

    
    elif op == "Resize Image":
        h, w = img.shape[:2]
        keep_ratio = st.checkbox("Preserve Aspect Ratio", True)
        new_w = st.number_input("New Width", 1, 4000, w)
        new_h = st.number_input("New Height", 1, 4000, h)

        if keep_ratio:
            ratio = w / h
            new_h = int(new_w / ratio)

        processed = cv2.resize(img, (int(new_w), int(new_h)))

    if processed is not None:
        st.subheader("Processed Image")
        st.image(processed, use_column_width=True)

        out_img = Image.fromarray(processed)
        buf = BytesIO()
        out_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Processed Image",
            data=byte_im,
            file_name="processed_image.png",
            mime="image/png"
        )


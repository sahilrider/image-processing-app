import streamlit as st
import numpy as np
import cv2

def image_thresholding():
    st.subheader("Image Thresholding")
    uploaded_file = st.file_uploader("Upload a image file", type=[".jpg", ".jpeg", ".png"])

    col1, col2 = st.beta_columns(2)

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # display original image
        col1.subheader("Original Image")
        col1.image(img)

    # types
    type_name = st.sidebar.selectbox("Choose thresholding type", ["Simple Thresholding",
                                                                  "Otsu Thresholding"])
    # params
    thresh1 = st.sidebar.slider('threshold1', 0, 255)
    thresh2 = st.sidebar.slider('threshold2', 0, 255)
    if (st.sidebar.button('Show Results')):
        if type_name == "Simple Thresholding":
            _, res = cv2.threshold(np.array(img), thresh1, thresh2, cv2.THRESH_BINARY)
        elif type_name == "Otsu Thresholding":
            _, res = cv2.threshold(np.array(img), thresh1, thresh2, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # display result
        col2.subheader(type_name)
        col2.image(res)


def morphological_transformation():
    st.subheader("Morphological Transformations")
    uploaded_file = st.file_uploader("Upload a image file", type=[".jpg", ".jpeg", ".png"])

    col1, col2 = st.beta_columns(2)

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # display original image
        col1.subheader("Original Image")
        col1.image(img)

    # types
    type_name = st.sidebar.selectbox("Choose thresholding type", ["Erosion",
                                                                  "Dilation",
                                                                  "Opening",
                                                                  "Closing",
                                                                  "Morphological Gradient"])
    # params
    k = st.sidebar.slider('kernel size', 1, 10)
    kernel = np.ones((k, k), np.uint8)
    if (st.sidebar.button('Show Results')):
        if type_name == "Erosion":
            res = cv2.erode(img, kernel, iterations=1)
        elif type_name == "Dilation":
            res = cv2.dilate(img, kernel, iterations=1)
        elif type_name == "Opening":
            res = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif type_name == "Closing":
            res = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        elif type_name == "Morphological Gradient":
            res = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        # display result
        col2.subheader(type_name)
        col2.image(res)


def canny():
    st.subheader("Canny Edge Detection")
    uploaded_file = st.file_uploader("Upload a image file", type=[".jpg", ".jpeg", ".png"])

    col1, col2 = st.beta_columns(2)

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        col1.subheader("Original Image")
        col1.image(img, channels="BGR")

    thresh1 = st.sidebar.slider('threshold1', 0, 255)
    thresh2 = st.sidebar.slider('threshold2', 0, 255)
    if st.sidebar.button('Detect Edges'):
        res = cv2.Canny(img, thresh1, thresh2)
        col2.subheader("Edges")
        col2.image(res)


def face_detection():
    st.subheader("Face Detection using Haar-cascade")
    uploaded_file = st.file_uploader("Upload a image file", type=[".jpg", ".jpeg", ".png"])

    col1, col2 = st.beta_columns(2)

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        col1.subheader("Original Image")
        col1.image(img, channels="BGR")

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if st.sidebar.button('Detect Faces'):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        col2.subheader("Detected Faces")
        col2.image(img, channels="BGR")

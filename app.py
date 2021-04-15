from streamlit_components import *


def main():
    st.header("Image Processing Demo with OpenCV")

    st.sidebar.title("Image Processing")
    function_selected = st.sidebar.selectbox("Choose OpenCV function",
                                             ["Image Thresholding",
                                              "Morphological Transformations",
                                              "Canny Edge Detection",
                                              "Face Detection"])
    if function_selected == "Image Thresholding":
        image_thresholding()
    elif function_selected == "Morphological Transformations":
        morphological_transformation()
    elif function_selected == "Canny Edge Detection":
        canny()
    elif function_selected == "Face Detection":
        face_detection()
    else:
        st.write("Choose right option")


if __name__ == "__main__":
    main()

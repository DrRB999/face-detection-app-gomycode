import cv2
import numpy as np
import streamlit as st


st.set_page_config(page_title="Face Detection App", page_icon="👤", layout="centered")

st.title("👤 Face Detection App — Viola-Jones")
st.markdown(
    """
    ### How to use the app
    1. Upload a **JPG, JPEG, or PNG** image.
    2. Choose the **rectangle color** used to highlight detected faces.
    3. Adjust **minNeighbors** to control detection strictness.
    4. Adjust **scaleFactor** to control the image pyramid scale.
    5. Download the processed image after face detection.
    """
)

# User-adjustable detection settings
rectangle_color = st.color_picker("Rectangle color", "#00FF00")
min_neighbors = st.slider("minNeighbors", min_value=1, max_value=10, value=5, step=1)
scale_factor = st.slider(
    "scaleFactor",
    min_value=1.01,
    max_value=2.00,
    value=1.10,
    step=0.01,
)

# Convert Streamlit HEX color to OpenCV BGR format
hex_color = rectangle_color.lstrip("#")
rgb_color = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

# Viola-Jones Haar cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("Could not load the Haar cascade classifier.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("The uploaded file could not be read as an image.")
        st.stop()

    # Convert to grayscale before Viola-Jones detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
    )

    processed_image = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(
            processed_image,
            (x, y),
            (x + w, y + h),
            bgr_color,
            2,
        )

    st.success(f"Detected faces: {len(faces)}")

    # Streamlit displays RGB images
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    st.image(processed_rgb, caption="Detected faces", use_container_width=True)

    # Explicit save with cv2.imwrite(), as requested by the checkpoint
    output_filename = "detected_faces.jpg"
    cv2.imwrite(output_filename, processed_image)

    # Download processed image to the user's device
    success, encoded_image = cv2.imencode(".jpg", processed_image)
    if success:
        st.download_button(
            "⬇️ Download detected image",
            data=encoded_image.tobytes(),
            file_name=output_filename,
            mime="image/jpeg",
        )

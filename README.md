# Face Detection App — Viola-Jones

A Streamlit application that detects faces in uploaded images using OpenCV's Viola-Jones Haar Cascade classifier.

## GoMyCode checkpoint requirements covered

- ✅ User instructions displayed in the Streamlit interface
- ✅ Save processed images with `cv2.imwrite()`
- ✅ Download processed images to the user's device
- ✅ Rectangle color selection with `st.color_picker()`
- ✅ Adjustable `minNeighbors` with `st.slider()`
- ✅ Adjustable `scaleFactor` with `st.slider()`
- ✅ Face detection with `face_cascade.detectMultiScale()`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project files

- `app.py` — Streamlit face-detection application
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation

## How it works

1. The user uploads an image.
2. The image is converted to grayscale with `cv2.cvtColor()`.
3. Viola-Jones detects faces with `detectMultiScale()`.
4. Rectangles are drawn around detected faces.
5. The processed image can be saved and downloaded.

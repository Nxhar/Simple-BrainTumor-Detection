import streamlit as st
import time
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.neighbors import KNeighborsClassifier

# Title of the app
st.title("Brain Tumor Detection")

# Uploading an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Function to train a simple KNN model
def train_knn():
    # Simulated dataset: areas and labels (0: small, 1: large)
    X = np.array([[1500], [1600], [2000], [2500], [3000], [3500],  [3600], [4000], [5500], [6000]])  # Bounding box areas
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])  # Labels: 0 for small, 1 for large

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

# Train KNN model
knn_model = train_knn()

if uploaded_file is not None:
    # Display the uploaded image with a specified width
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Button to get results
    if st.button("Get Results"):
        # Simulate processing time
        with st.spinner("Processing..."):
            time.sleep(1)  # Simulating a delay for processing

        # Load the YOLOv8 model
        model = YOLO('tumordetect.pt')

        # Convert the uploaded image to an OpenCV image
        image_np = np.array(uploaded_image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Perform inference
        results = model(image_cv)

        # Extract results
        result_image = image_cv.copy()
        for result in results:
            for box in result.boxes:
                # Extract bounding box and other information
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Calculate area of the bounding box
                area = (x_max - x_min) * (y_max - y_min)

                # Predict tumor size using KNN
                tumor_size_label = knn_model.predict([[area]])[0]
                tumor_size = "Large" if tumor_size_label == 1 else "Small"

                # Draw bounding box and label on the image
                cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(result_image, f"{class_name} ({confidence:.2f})", (x_min, y_min - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.putText(result_image, f"Tumor Size: {tumor_size}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Convert image back to RGB
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        # Display the output image
        st.image(result_image_rgb, caption="Detection Result", use_column_width=True)

        # Display prediction results
        st.write("### Prediction Results:")
        none = True
        for result in results:
            for box in result.boxes:
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

                # Calculate area of the bounding box
                area = (x_max - x_min) * (y_max - y_min)

                # Predict tumor size using KNN
                tumor_size_label = knn_model.predict([[area]])[0]
                tumor_size = "Large" if tumor_size_label == 1 else "Small"

                st.write(f"- **Class:** {class_name}")
                st.write(f"- **Confidence:** {confidence:.2f}")
                st.write(f"- **Bounding Box:** (xmin: {x_min}, ymin: {y_min}, xmax: {x_max}, ymax: {y_max})")
                st.write(f"- **Tumor Size:** {tumor_size}")

                none = False
        if none == True:
            st.write(f"**No Tumor Detected!**")
        
else:
    st.write("Please upload an image to get started.")

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import torch
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Streamlit UI
st.set_page_config(page_title="Fast Intruder Detection", layout="centered", page_icon="⚡")
st.title("⚡ Fast Intruder Detection with MediaPipe + DeepFace")
st.subheader("Real-Time Face Detection (MediaPipe) + Recognition (DeepFace)")

# Sidebar controls
camera_source = st.sidebar.selectbox("Select Camera", options=["Webcam (0)", "RTSP"], index=0)
rtsp_url = st.sidebar.text_input("RTSP URL", value="rtsp://") if camera_source == "RTSP" else None
model_name = st.sidebar.selectbox("Embedding Model", options=["SFace", "Facenet", "ArcFace", "OpenFace", "DeepID", "Dlib"], index=0)
threshold = st.sidebar.slider("Cosine Similarity Threshold", 0.0, 1.0, 0.65, 0.01)
run_button = st.sidebar.button("Start Detection")

# Show CPU/GPU info
st.sidebar.markdown(f"**CUDA Available (Torch):** {torch.cuda.is_available()}")
if torch.cuda.is_available():
    st.sidebar.markdown(f"**GPU:** {torch.cuda.get_device_name(0)}")

# Load known face embeddings using selected model
@st.cache_resource
def load_known_embeddings(model_name, directory="known_faces"):
    db = {}
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        try:
            embedding = DeepFace.represent(img_path=path, model_name=model_name, enforce_detection=False)[0]['embedding']
            db[os.path.splitext(file)[0]] = np.array(embedding)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    return db

known_db = load_known_embeddings(model_name)

# Alert cooldown
ALERT_COOLDOWN = 10
last_alert_time = datetime.now()

# MediaPipe face detection setup
mp_face_detection = mp.solutions.face_detection

# Start detection
if run_button:
    st.write(f"Running MediaPipe + DeepFace ({model_name}) intruder detection...")
    source = rtsp_url if camera_source == "RTSP" else 0
    cap = cv2.VideoCapture(source)
    placeholder = st.empty()
    unknown_count = 0

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)

            if results.detections:
                h, w, _ = frame.shape
                for detection in results.detections:
                    box = detection.location_data.relative_bounding_box
                    x1 = int(box.xmin * w)
                    y1 = int(box.ymin * h)
                    x2 = int((box.xmin + box.width) * w)
                    y2 = int((box.ymin + box.height) * h)

                    face_crop = rgb[y1:y2, x1:x2]
                    try:
                        emb = DeepFace.represent(img_path=face_crop, model_name=model_name, enforce_detection=False)[0]['embedding']
                        emb = np.array(emb)
                        best_name = "Unknown"
                        best_sim = 0.0

                        for name, ref_emb in known_db.items():
                            sim = 1 - cosine(emb, ref_emb)
                            if sim > best_sim:
                                best_sim = sim
                                best_name = name

                        if best_sim >= threshold:
                            label = f"{best_name} ({best_sim:.2f})"
                            color = (0, 255, 0)
                        else:
                            label = "🚨 Unknown"
                            color = (0, 0, 255)
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            os.makedirs("intruders", exist_ok=True)
                            cv2.imwrite(f"intruders/{timestamp}.jpg", frame)

                            current_time = datetime.now()
                            if (current_time - last_alert_time).total_seconds() > ALERT_COOLDOWN:
                                last_alert_time = current_time
                                st.warning(f"🚨 Unknown face detected at {timestamp}")

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    except Exception as e:
                        print(f"Recognition error: {e}")

            placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    placeholder.empty()
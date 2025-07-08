import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import torch
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Page setup
st.set_page_config(page_title="⚡ Intruder Detection Demo", layout="wide", page_icon="🛡️")
st.title("🛡️ Intruder Detector Demo")

# Initialize session state
if "known_db" not in st.session_state:
    st.session_state.known_db = {}
if "run_detection" not in st.session_state:
    st.session_state.run_detection = False

# Sidebar — Known Face Management
with st.sidebar:
    st.header("📷 Add Known Faces")

    uploaded_photo = st.camera_input("Take a Known Face Photo from Multiple Angle")

    if uploaded_photo:
        name_input = st.text_input("Name for this Face", value="", key="face_name")
        if st.button("➕ Add Face"):
            if not name_input.strip():
                st.error("Please enter a name.")
            else:
                img_bytes = uploaded_photo.getvalue()
                np_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                try:
                    embedding = DeepFace.represent(img_path=np_img, model_name="SFace", enforce_detection=False)[0]['embedding']
                    st.session_state.known_db[name_input.strip()] = np.array(embedding)
                    st.success(f"Added known face: {name_input.strip()}")
                except Exception as e:
                    st.error(f"Error processing photo: {e}")

    if st.session_state.known_db:
        with st.expander("👥 View Known Faces"):
            for name in st.session_state.known_db.keys():
                st.markdown(f"- {name}")
    else:
        st.info("No known faces added yet.")

# Sidebar — Detection Controls
with st.sidebar:
    st.header("🎛️ Detection Settings")
    camera_source = st.selectbox("Camera Source", ["Webcam (0)", "RTSP"])
    rtsp_url = st.text_input("RTSP URL", "rtsp://") if camera_source == "RTSP" else None
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.24, 0.01)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Detection"):
            st.session_state.run_detection = True
    with col2:
        if st.button("⏹️ Stop Detection"):
            st.session_state.run_detection = False

# Alert cooldown
ALERT_COOLDOWN = 10
last_alert_time = datetime.now()

# MediaPipe face detection setup
mp_face_detection = mp.solutions.face_detection

# Run detection loop
if st.session_state.run_detection:
    if not st.session_state.known_db:
        st.error("⚠️ Please add at least one known face before starting detection.")
    else:
        st.success("✅ Detection Running...")
        source = rtsp_url if camera_source == "RTSP" else 0
        cap = cv2.VideoCapture(source)
        frame_display = st.empty()

        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detector:
            while cap.isOpened() and st.session_state.run_detection:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read frame.")
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
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        face_crop = rgb[y1:y2, x1:x2]

                        if face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
                            continue

                        try:
                            face_crop = cv2.resize(face_crop, (112, 112))
                            emb = DeepFace.represent(img_path=face_crop, model_name="SFace", enforce_detection=False)[0]['embedding']
                            emb = np.array(emb)
                            best_name = "Unknown"
                            best_sim = 0.0

                            for name, ref_emb in st.session_state.known_db.items():
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
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                current_time = datetime.now()
                                if (current_time - last_alert_time).total_seconds() > ALERT_COOLDOWN:
                                    last_alert_time = current_time
                                    st.warning(f"🚨 Unknown face detected at {timestamp}")

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        except Exception as e:
                            print(f"Recognition error: {e}")

                frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        frame_display.empty()
        st.success("🛑 Detection Stopped.")

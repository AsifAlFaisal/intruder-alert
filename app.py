import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2, numpy as np, av
import mediapipe as mp
from deepface import DeepFace
from scipy.spatial.distance import cosine
from datetime import datetime

# Page setup
st.set_page_config(page_title="⚡ Live Intruder Detector", layout="wide")
st.title("🛡️ Live Intruder Detector")

# Known faces storage
if "known_db" not in st.session_state:
    st.session_state.known_db = {}

# Sidebar: add known faces
with st.sidebar:
    st.header("📷 Add Known Faces")
    img_file = st.camera_input("Take a photo")
    if img_file:
        name = st.text_input("Name", key="face_name")
        if st.button("➕ Add Face"):
            img = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = DeepFace.represent(img_path=img, model_name="SFace", enforce_detection=False)[0]['embedding']
            st.session_state.known_db[name.strip()] = np.array(emb)
            st.success(f"Added: {name}")

    st.header("👥 Database")
    if st.session_state.known_db:
        for n in st.session_state.known_db:
            st.write("-", n)
    else:
        st.info("No known faces yet.")

# Sidebar: detection settings
threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.24, 0.01)

# Setup MediaPipe face detection
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# WebRTC config for browser webcam
RTC_CONF = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Alert cooldown
ALERT_COOLDOWN = 10  # seconds
last_alert = datetime.now()

# Processor that works on each live frame
class IntruderDetector(VideoTransformerBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global last_alert
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)

        if results.detections:
            h, w, _ = img.shape
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                x1, y1 = int(bb.xmin * w), int(bb.ymin * h)
                x2, y2 = x1 + int(bb.width * w), y1 + int(bb.height * h)
                x1, y1 = max(0, x1), max(0, y1)
                if y2 - y1 < 50 or x2 - x1 < 50:
                    continue

                face = cv2.resize(rgb[y1:y2, x1:x2], (112, 112))
                emb = np.array(
                    DeepFace.represent(img_path=face, model_name="SFace", enforce_detection=False)[0]["embedding"]
                )

                name, best = "Unknown", 0.0
                for nm, ref in st.session_state.known_db.items():
                    sim = 1 - cosine(emb, ref)
                    if sim > best:
                        best, name = sim, nm

                if best >= threshold:
                    label, color = f"{name} ({best:.2f})", (0, 255, 0)
                else:
                    label, color = "🚨 Unknown", (0, 0, 255)
                    now = datetime.now()
                    if (now - last_alert).total_seconds() > ALERT_COOLDOWN:
                        last_alert = now
                        st.warning(f"🚨 Unknown detected at {now:%Y-%m-%d %H:%M:%S}")

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from deepface import DeepFace
from scipy.spatial.distance import cosine
import av

# Page setup
st.set_page_config(page_title="⚡ Intruder Detection Demo", layout="wide", page_icon="🛡️")
st.title("🛡️ Intruder Detector Demo")

# Session state
if "known_db" not in st.session_state:
    st.session_state.known_db = {}
if "run_detection" not in st.session_state:
    st.session_state.run_detection = False

# Sidebar — face management
with st.sidebar:
    st.header("📷 Add Known Faces")
    uploaded = st.camera_input("Take a Known Face Photo")
    if uploaded:
        name = st.text_input("Name", key="face_name")
        if st.button("➕ Add Face"):
            img = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = DeepFace.represent(img_path=img, model_name="SFace", enforce_detection=False)[0]['embedding']
            st.session_state.known_db[name.strip()] = np.array(emb)
            st.success(f"Added: {name}")

    if st.session_state.known_db:
        st.expander("👥 Known Faces", expanded=True)
        for n in st.session_state.known_db:
            st.write(f"- {n}")
    else:
        st.info("No known faces yet.")

# Sidebar — controls
with st.sidebar:
    st.header("🎛️ Settings")
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.24, 0.01)
    if st.button("▶️ Start"):
        st.session_state.run_detection = True
    if st.button("⏹️ Stop"):
        st.session_state.run_detection = False

# MediaPipe setup
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# WebRTC config
RTC_CONF = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})

# Context and alert state
ALERT_COOLDOWN = 10
last_alert = datetime.now()

# Define processor
class IntruderProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global last_alert
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)

        if results.detections:
            h, w, _ = img.shape
            for d in results.detections:
                box = d.location_data.relative_bounding_box
                x1, y1 = int(box.xmin*w), int(box.ymin*h)
                x2, y2 = x1 + int(box.width*w), y1 + int(box.height*h)
                x1, y1 = max(0, x1), max(0, y1)
                face = cv2.resize(rgb[y1:y2, x1:x2], (112,112))
                emb = np.array(DeepFace.represent(img_path=face, model_name="SFace", enforce_detection=False)[0]['embedding'])

                best_name, best_sim = "Unknown", 0
                for nm, ref in st.session_state.known_db.items():
                    sim = 1 - cosine(emb, ref)
                    if sim > best_sim:
                        best_name, best_sim = nm, sim

                if best_sim >= threshold:
                    label, color = f"{best_name} ({best_sim:.2f})", (0,255,0)
                else:
                    label, color = "🚨 Unknown", (0,0,255)
                    now = datetime.now()
                    if (now - last_alert).total_seconds() > ALERT_COOLDOWN:
                        last_alert = now
                        st.warning(f"🚨 Unknown detected @ {now:%Y-%m-%d %H:%M:%S}")

                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start WebRTC streamer
if st.session_state.run_detection:
    webrtc_ctx = webrtc_streamer(
        key="intruder",
        video_processor_factory=IntruderProcessor,
        rtc_configuration=RTC_CONF,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

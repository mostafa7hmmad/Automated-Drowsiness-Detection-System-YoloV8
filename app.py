import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from streamlit.components.v1 import html
import av

# إعداد WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# تحميل موديل YOLO (تأكد إن paste.pt موجود في نفس مجلد المشروع)
model = YOLO("best.pt")
model.fuse()

# أسماء التصنيفات وألوانها
CLASS_NAMES = ["microsleep", "neutral", "yawning"]
COLOR_MAP = {
    "microsleep": (0, 0, 255),
    "neutral": (0, 255, 0),
    "yawning": (0, 0, 255)
}

# رابط الصوت من GitHub (بدّله برابطك المباشر)
BUZZER_URL = "https://raw.githubusercontent.com/mostafa7hmmad/yolov8-drowsiness-detection-system/main/1.mp3"

# تشغيل البازر (في المتصفح)
def play_buzzer():
    html(f"""
        <audio id="buzzer" autoplay loop>
            <source src="{BUZZER_URL}" type="audio/mpeg">
        </audio>
        <script>
            var buzzer = document.getElementById("buzzer");
            if (buzzer) buzzer.play();
        </script>
    """, height=0)

# إيقاف البازر
def stop_buzzer():
    html("""
        <script>
            var buzzer = document.getElementById("buzzer");
            if (buzzer) buzzer.pause();
        </script>
    """, height=0)

# معالج الفيديو
class FastVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_skip = 2
        self.counter = 0
        self.prev_result = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        small_frame = cv2.resize(img, (320, 240))

        self.counter += 1
        if self.counter % self.frame_skip == 0:
            results = model(small_frame, verbose=False)[0]
            self.prev_result = results
        else:
            results = self.prev_result

        detected_non_neutral = False

        if results:
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label_index = int(cls)
                if label_index >= len(CLASS_NAMES): continue
                label = CLASS_NAMES[label_index]
                color = COLOR_MAP[label]
                x1, y1, x2, y2 = [int(x * 2) for x in (x1, y1, x2, y2)]

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                if label != "neutral":
                    detected_non_neutral = True

        # حفظ حالة البازر
        st.session_state['play_buzzer'] = detected_non_neutral
        return img

# Streamlit واجهة
st.set_page_config(page_title="YOLOv8 Drowsiness Detection", layout="wide")
st.title("🧠 Real-time Drowsiness Detection with Sound Alert")

# زر ملء الشاشة
st.markdown("""
<button onclick="document.querySelector('video').requestFullscreen()" style="
    display:block;
    margin:auto;
    background-color:#0a84ff;
    color:white;
    padding:10px 30px;
    border-radius:12px;
    font-size:16px;
    border:none;
    cursor:pointer;">📺 Fullscreen Camera</button>
""", unsafe_allow_html=True)

# تهيئة الحالة
if "play_buzzer" not in st.session_state:
    st.session_state["play_buzzer"] = False

# الكاميرا
webrtc_streamer(
    key="fast-stream",
    video_processor_factory=FastVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    async_processing=True
)

# تشغيل أو إيقاف البازر بناءً على الحالة
if st.session_state["play_buzzer"]:
    play_buzzer()
else:
    stop_buzzer()

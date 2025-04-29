import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# إعداد STUN للسيرفر WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# تحميل موديل YOLO
model = YOLO("best.pt")  # تأكد أن هذا الملف موجود في نفس المسار

# خريطة التصنيفات
CLASS_NAMES = ["microsleep", "neutral", "yawning"]
COLOR_MAP = {
    "microsleep": (0, 0, 255),  # أحمر
    "neutral": (0, 255, 0),     # أخضر
    "yawning": (0, 0, 255)      # أحمر
}

# معالج الفيديو
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, verbose=False)[0]

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label_index = int(cls)
            label = CLASS_NAMES[label_index]
            color = COLOR_MAP[label]

            # رسم المستطيل والنص
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )
        return img

# واجهة Streamlit
st.set_page_config(
    page_title="Real-Time Drowsiness Detection",
    layout="wide"
)
st.markdown("<h1 style='text-align: center;'>🚨 Real-Time Face State Detection (YOLOv8)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detects Microsleep, Neutral, and Yawning with webcam. Toggle fullscreen below.</p>", unsafe_allow_html=True)

# زر تكبير الكاميرا
st.markdown("""
    <style>
    .fullscreen-btn {
        display: block;
        margin: 10px auto;
        padding: 10px 25px;
        font-size: 18px;
        background-color: #0a84ff;
        color: white;
        border: none;
        border-radius: 10px;
        cursor: pointer;
    }
    </style>
    <button class="fullscreen-btn" onclick="document.querySelector('video').requestFullscreen()">🔍 Fullscreen Camera</button>
    <script>
        document.addEventListener('fullscreenchange', () => {
            if (!document.fullscreenElement) {
                console.log('Exited fullscreen');
            }
        });
    </script>
""", unsafe_allow_html=True)

# بث الفيديو
webrtc_streamer(
    key="live-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# إعداد RTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# تحميل موديل YOLO nano لسرعة أعلى
model = YOLO("yolov8n.pt")  # استخدم موديل خفيف للتجربة
model.fuse()  # لتسريع الـ inference

# أسماء التصنيفات واللون
CLASS_NAMES = ["microsleep", "neutral", "yawning"]
COLOR_MAP = {
    "microsleep": (0, 0, 255),
    "neutral": (0, 255, 0),
    "yawning": (0, 0, 255)
}

# معالج الفيديو
class FastVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_skip = 2  # نفذ على كل ثاني Frame
        self.counter = 0
        self.prev_result = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # تقليل الريزولوشن قبل المعالجة
        small_frame = cv2.resize(img, (320, 240))

        self.counter += 1
        if self.counter % self.frame_skip == 0:
            results = model(small_frame, verbose=False)[0]
            self.prev_result = results
        else:
            results = self.prev_result

        if results:
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label_index = int(cls)
                if label_index >= len(CLASS_NAMES): continue
                label = CLASS_NAMES[label_index]
                color = COLOR_MAP[label]

                # تحجيم البوكس بناءً على الفرق في الريزولوشن
                x1, y1, x2, y2 = [int(x * 2) for x in (x1, y1, x2, y2)]

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return img

# Streamlit App
st.set_page_config(page_title="Fast Drowsiness Detection", layout="wide")
st.title("🚀 Fast YOLOv8 Live Detection")

# زر تكبير الشاشة
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
    cursor:pointer;">Fullscreen Camera</button>
""", unsafe_allow_html=True)

# تشغيل الكاميرا
webrtc_streamer(
    key="fast-stream",
    video_processor_factory=FastVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    async_processing=True
)

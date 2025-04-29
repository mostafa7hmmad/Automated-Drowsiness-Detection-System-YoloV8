import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Configure STUN server for WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Load your trained YOLOv8 model (replace 'best.pt' with your model path)
model = YOLO("best.pt")

# Video frame processor for detection
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.buzzer_on = False

    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        # Run YOLOv8 inference
        results = model(img)[0]
        not_natural_detected = False

        # Iterate detections
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            color = (0, 255, 0) if label == "Natural" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if label != "Natural":
                not_natural_detected = True

        # Immediate buzzer logic
        self.buzzer_on = not_natural_detected

        return img

# Initialize processor
processor = VideoProcessor()

# Create Streamlit UI
st.title("YOLOv8 Face Naturality Detection with Eye Monitoring")
st.write("إذا تم تصنيف الحالة كـ 'Not Natural' (عيون مغلقة)، يصدر صوت تنبيه فوري حتى تعود الحالة لـ 'Natural'.")

# Start webcam stream
webrtc_ctx = webrtc_streamer(
    key="yolo-stream",
    video_processor_factory=lambda: processor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Play alert sound immediately when Not Natural
if webrtc_ctx.state.playing and processor.buzzer_on:
    st.audio("alert.mp3", format="audio/mp3", start_time=0)

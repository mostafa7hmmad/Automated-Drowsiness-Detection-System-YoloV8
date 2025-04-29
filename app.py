import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import time

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
        self.eye_closed_start = None

    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        # Run YOLOv8 inference
        results = model(img)[0]
        natural_detected = False
        not_natural_detected = False
        eyes_closed_detected = False

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            color = (0, 255, 0) if label == "Natural" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if label == "Natural":
                natural_detected = True
            else:
                not_natural_detected = True

            # Optional: use bounding box size/location heuristics to identify eyes
            # Here we assume "Not Natural" also tags closed eyes – simplify for now

        # Detect if eyes are closed longer than 3 seconds
        if not_natural_detected:
            if self.eye_closed_start is None:
                self.eye_closed_start = time.time()
            elif time.time() - self.eye_closed_start >= 3:
                eyes_closed_detected = True
        else:
            self.eye_closed_start = None

        # Determine buzzer logic
        if not_natural_detected or eyes_closed_detected:
            self.buzzer_on = True
        elif natural_detected:
            self.buzzer_on = False

        return img

# Initialize processor
processor = VideoProcessor()

# Create Streamlit UI
st.title("YOLOv8 Face Naturality Detection with Eye Monitoring")
st.write("إذا تم تصنيف الحالة كـ 'Not Natural' أو تم إغلاق العين أكثر من 3 ثوانٍ، سيتم إصدار صوت تنبيه حتى تصبح الحالة 'Natural'.")

# Start webcam stream
webrtc_ctx = webrtc_streamer(
    key="yolo-stream",
    video_processor_factory=lambda: processor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Play buzzer sound if needed
if webrtc_ctx.state.playing:
    if processor.buzzer_on:
        st.markdown(
            "<audio autoplay><source src='alert.mp3' type='audio/wav'></audio>",
            unsafe_allow_html=True
        )

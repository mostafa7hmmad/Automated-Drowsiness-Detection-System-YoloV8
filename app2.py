from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
CORS(app)

# Load YOLOv8 model
model = YOLO("best.pt")  # تأكد إن ملف best.pt في نفس المكان

@app.route("/predict", methods=["POST"])
def predict():
    if "frame" not in request.files:
        return jsonify([])

    file = request.files["frame"].read()
    nparr = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)[0]
    response = []

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box.tolist())
        label = model.names[int(cls)]
        response.append({
            "label": label,
            "bbox": [x1, y1, x2, y2]
        })

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const buzzer = document.getElementById("buzzer");

// 🧠 استبدل بـ API حقيقية أو داتا وهمية للتجريب
async function getDetections(frame) {
  // دا مجرد داتا تجريبية - غيره لما تستخدم موديلك
  return [
    { bbox: [100, 80, 300, 280], label: "microsleep" },
    { bbox: [400, 100, 600, 300], label: "neutral" }
  ];
}

function drawBoxes(detections) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  let foundNonNeutral = false;

  detections.forEach(d => {
    const [x1, y1, x2, y2] = d.bbox;
    const label = d.label;

    if (label !== "neutral") foundNonNeutral = true;

    ctx.strokeStyle = label === "neutral" ? "lime" : "red";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.fillRect(x1, y1 - 25, ctx.measureText(label).width + 20, 25);

    ctx.fillStyle = "#fff";
    ctx.font = "16px Arial";
    ctx.fillText(label, x1 + 5, y1 - 7);
  });

  // 🔊 البازر
  if (foundNonNeutral) {
    if (buzzer.paused) buzzer.play();
  } else {
    if (!buzzer.paused) {
      buzzer.pause();
      buzzer.currentTime = 0;
    }
  }
}

async function detectLoop() {
  if (video.readyState === video.HAVE_ENOUGH_DATA) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const detections = await getDetections(canvas); // simulate
    drawBoxes(detections);
  }

  requestAnimationFrame(detectLoop);
}

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play();
      detectLoop();
    };
  })
  .catch(err => {
    console.error("Error accessing webcam:", err);
  });

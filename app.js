// SpeakFlow — client-side face tracking + session logic.
// MediaPipe runs entirely in-browser via WASM, loaded from CDN below.
// No video/images ever leave the browser — only small numeric summaries
// are sent to /api/feedback at the end of a session.

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const MAX_SESSION_SECONDS = 120;
const BLINK_THRESHOLD = 0.5; // blendshape score above which an eye counts as "closed"

// ---- DOM references ----
const screens = {
  landing: document.getElementById("landing"),
  session: document.getElementById("session"),
  results: document.getElementById("results"),
};
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const restartBtn = document.getElementById("restartBtn");
const landingError = document.getElementById("landingError");
const sessionStatus = document.getElementById("sessionStatus");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");

const timerValueEl = document.getElementById("timerValue");
const blinkValueEl = document.getElementById("blinkValue");
const gazeValueEl = document.getElementById("gazeValue");
const postureValueEl = document.getElementById("postureValue");

// ---- Session state ----
let faceLandmarker = null;
let stream = null;
let rafId = null;
let sessionStartMs = 0;
let timerIntervalId = null;

let blinkCount = 0;
let eyesCurrentlyClosed = false;
let gazeSamples = []; // 0-1 deviation-from-center per frame
let headAngleSamples = []; // { yaw, pitch, roll } per frame, degrees-ish

function showScreen(name) {
  Object.values(screens).forEach((el) => el.classList.add("hidden"));
  screens[name].classList.remove("hidden");
}

// ---- Load the face landmarker model (once) ----
async function loadFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false,
    runningMode: "VIDEO",
    numFaces: 1,
  });
}

// ---- Start a session ----
async function startSession() {
  landingError.textContent = "";
  startBtn.disabled = true;

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });
  } catch (err) {
    landingError.textContent =
      "Couldn't access your camera. Please allow camera permission and try again.";
    startBtn.disabled = false;
    return;
  }

  video.srcObject = stream;
  showScreen("session");
  sessionStatus.textContent = "Loading face tracking model…";

  if (!faceLandmarker) {
    try {
      await loadFaceLandmarker();
    } catch (err) {
      sessionStatus.textContent =
        "Failed to load face tracking. Check your connection and reload the page.";
      return;
    }
  }

  // Reset session data
  blinkCount = 0;
  eyesCurrentlyClosed = false;
  gazeSamples = [];
  headAngleSamples = [];
  sessionStartMs = performance.now();

  sessionStatus.textContent = "Tracking your face — look at the camera and speak naturally.";
  stopBtn.classList.remove("hidden");

  video.addEventListener("loadeddata", () => {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    detectFrame();
  }, { once: true });

  timerIntervalId = setInterval(updateTimerDisplay, 250);
}

function updateTimerDisplay() {
  const elapsedSec = (performance.now() - sessionStartMs) / 1000;
  const m = Math.floor(elapsedSec / 60);
  const s = Math.floor(elapsedSec % 60);
  timerValueEl.textContent = `${m}:${s.toString().padStart(2, "0")}`;

  if (elapsedSec >= MAX_SESSION_SECONDS) {
    endSession();
  }
}

// ---- Per-frame detection loop ----
function detectFrame() {
  if (!faceLandmarker || video.readyState < 2) {
    rafId = requestAnimationFrame(detectFrame);
    return;
  }

  const result = faceLandmarker.detectForVideo(video, performance.now());
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  if (result.faceLandmarks && result.faceLandmarks.length > 0) {
    const landmarks = result.faceLandmarks[0];
    const blendshapes = result.faceBlendshapes?.[0]?.categories ?? [];

    processBlink(blendshapes);
    processGaze(blendshapes);
    processHeadAngle(landmarks);
    drawSimpleOverlay(landmarks);
  }

  rafId = requestAnimationFrame(detectFrame);
}

function getBlendshapeScore(categories, name) {
  const found = categories.find((c) => c.categoryName === name);
  return found ? found.score : 0;
}

function processBlink(blendshapes) {
  const left = getBlendshapeScore(blendshapes, "eyeBlinkLeft");
  const right = getBlendshapeScore(blendshapes, "eyeBlinkRight");
  const avgBlink = (left + right) / 2;

  if (avgBlink > BLINK_THRESHOLD && !eyesCurrentlyClosed) {
    eyesCurrentlyClosed = true;
    blinkCount += 1;
    blinkValueEl.textContent = String(blinkCount);
  } else if (avgBlink <= BLINK_THRESHOLD && eyesCurrentlyClosed) {
    eyesCurrentlyClosed = false;
  }
}

function processGaze(blendshapes) {
  // Higher score on any of these = eyes looking away from center.
  const lookAwayNames = [
    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
  ];
  const maxDeviation = Math.max(
    ...lookAwayNames.map((n) => getBlendshapeScore(blendshapes, n)),
    0
  );
  gazeSamples.push(maxDeviation);

  // Show a live rolling gaze indicator
  const recentAvg = average(gazeSamples.slice(-30));
  gazeValueEl.textContent = recentAvg < 0.25 ? "Steady" : recentAvg < 0.5 ? "Drifting" : "Away";
}

function processHeadAngle(landmarks) {
  // Approximate head yaw/pitch/roll from landmark positions.
  // Indices: 1 = nose tip, 33 = right eye outer corner, 263 = left eye outer corner.
  const nose = landmarks[1];
  const rightEye = landmarks[33];
  const leftEye = landmarks[263];

  const midX = (leftEye.x + rightEye.x) / 2;
  const midY = (leftEye.y + rightEye.y) / 2;
  const interEyeDist = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y) || 0.001;

  const yaw = ((nose.x - midX) / interEyeDist) * 100;
  const pitch = ((nose.y - midY) / interEyeDist) * 100;
  const roll =
    (Math.atan2(leftEye.y - rightEye.y, leftEye.x - rightEye.x) * 180) / Math.PI;

  headAngleSamples.push({ yaw, pitch, roll });

  const recent = headAngleSamples.slice(-30);
  const tiltVariance = variance(recent.map((h) => h.roll)) + variance(recent.map((h) => h.yaw));
  postureValueEl.textContent = tiltVariance < 15 ? "Stable" : tiltVariance < 40 ? "Shifting" : "Restless";
}

function drawSimpleOverlay(landmarks) {
  // Minimal visual: small dots on eyes + nose so the user sees tracking is live.
  const pointsToDraw = [1, 33, 263, 133, 362]; // nose, eye corners, eye inner corners
  overlayCtx.fillStyle = "#4f7cff";
  pointsToDraw.forEach((i) => {
    const p = landmarks[i];
    if (!p) return;
    const x = p.x * overlay.width;
    const y = p.y * overlay.height;
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, 3, 0, Math.PI * 2);
    overlayCtx.fill();
  });
}

function average(arr) {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function variance(arr) {
  if (arr.length === 0) return 0;
  const m = average(arr);
  return average(arr.map((v) => (v - m) ** 2));
}

// ---- End session, compute summary, get AI feedback ----
async function endSession() {
  if (rafId) cancelAnimationFrame(rafId);
  if (timerIntervalId) clearInterval(timerIntervalId);
  if (stream) stream.getTracks().forEach((t) => t.stop());

  const durationSec = (performance.now() - sessionStartMs) / 1000;
  const blinkRatePerMin = durationSec > 0 ? (blinkCount / durationSec) * 60 : 0;

  const avgGazeDeviation = average(gazeSamples);
  const gazeStabilityScore = Math.round(Math.max(0, 100 - avgGazeDeviation * 100));

  const rollVariance = variance(headAngleSamples.map((h) => h.roll));
  const yawVariance = variance(headAngleSamples.map((h) => h.yaw));
  const postureStabilityScore = Math.round(
    Math.max(0, 100 - (rollVariance + yawVariance) * 1.5)
  );

  const summary = {
    durationSec: Math.round(durationSec),
    blinkCount,
    blinkRatePerMin: Math.round(blinkRatePerMin),
    gazeStabilityScore,
    postureStabilityScore,
  };

  showResults(summary);
  fetchAiFeedback(summary);
}

function showResults(summary) {
  showScreen("results");
  document.getElementById("resDuration").textContent = `${summary.durationSec}s`;
  document.getElementById("resBlinkRate").textContent = summary.blinkRatePerMin;
  document.getElementById("resGaze").textContent = `${summary.gazeStabilityScore}%`;
  document.getElementById("resPosture").textContent = `${summary.postureStabilityScore}%`;
  document.getElementById("feedbackText").textContent = "Generating feedback…";
}

async function fetchAiFeedback(summary) {
  try {
    const res = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(summary),
    });
    const data = await res.json();
    document.getElementById("feedbackText").textContent =
      data.feedback || "Couldn't generate feedback this time — but your stats above are real.";
  } catch (err) {
    document.getElementById("feedbackText").textContent =
      "Couldn't reach the feedback service. Your session stats above are still accurate.";
  }
}

function resetToLanding() {
  showScreen("landing");
  startBtn.disabled = false;
  stopBtn.classList.add("hidden");
}

// ---- Wire up buttons ----
startBtn.addEventListener("click", startSession);
stopBtn.addEventListener("click", endSession);
restartBtn.addEventListener("click", resetToLanding);

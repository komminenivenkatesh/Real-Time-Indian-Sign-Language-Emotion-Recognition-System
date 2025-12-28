import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
from collections import deque, Counter
import time

# ---------------- Paths ----------------
MODEL_PATH = "data/models/landmarks_model.pkl"  # classic sklearn model (not the LSTM .pt)
ENC_PATH = "data/models/label_encoder.pkl"

# ---------------- Load model ------------
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENC_PATH)

# ---------------- TTS -------------------
engine = pyttsx3.init()
engine.setProperty("rate", 165)

SAY_MAP = {
    str(i): s for i, s in enumerate(["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"])
}

# ---------------- MediaPipe -------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ---------------- Helpers ----------------
def normalize_hand(arr21x3):
    """
    Translate to wrist=0, and scale-invariant normalize.
    If no landmarks detected (all zeros), return zeros as-is.
    """
    if np.allclose(arr21x3, 0):
        return arr21x3
    # translate
    arr = arr21x3.copy()
    arr -= arr[0]  # wrist as origin
    # scale: use RMS distance to wrist to avoid tiny scales
    scale = np.sqrt((arr**2).sum(axis=1)).mean()
    if scale < 1e-6:
        return arr
    arr /= scale
    return arr


def predict_with_conf(x_feat):
    """
    x_feat: (1, D)
    returns (label_str, conf_float)
    """
    # try probability if available
    conf = 1.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_feat)[0]
        idx = int(np.argmax(probs))
        conf = float(np.max(probs))
    else:
        # Some models (e.g., SVC without probability) don't provide proba
        idx = int(model.predict(x_feat)[0])
    label = label_encoder.inverse_transform([idx])[0]
    return label, conf


# ---------------- Webcam -----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("⚠️ No webcam found!")
    exit()

print("✅ Webcam started. Press 'q' to quit.")

# ---------------- Smoothing/Thresholds ---------------
SMOOTH_N = 7  # majority vote window
CONF_THRESH = 0.70  # require this confidence to show/speak
VOTE_MIN_FRAC = 0.6  # majority fraction inside window
last_spoken = ""  # what we last said
pred_history = deque(maxlen=SMOOTH_N)
last_say_time = 0.0
SAY_COOLDOWN = 0.8  # seconds between TTS messages

# FPS
t0 = time.time()
frame_count = 0
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # default UI text
    shown_label = "Detecting..."
    shown_conf = 0.0

    # Prepare landmarks
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        # Draw + split into left/right consistently
        for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            handed = hd.classification[0].label  # "Left" or "Right"
            if handed == "Left":
                left = coords
            else:
                right = coords
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # Normalize
        left_n = normalize_hand(left)
        right_n = normalize_hand(right)

        # Feature: concat both (keeps shape 126 even if a hand is missing)
        feat = np.concatenate([left_n.flatten(), right_n.flatten()])[None, :]  # (1,126)

        # Predict
        label_raw, conf = predict_with_conf(feat)

        # Smooth by vote
        pred_history.append((label_raw, conf))
        labels_only = [p[0] for p in pred_history]
        most, count = Counter(labels_only).most_common(1)[0]
        frac = count / len(labels_only)

        # Accept only if both conditions pass
        if conf >= CONF_THRESH and frac >= VOTE_MIN_FRAC:
            shown_label = most
            shown_conf = conf
        else:
            shown_label = "Detecting..."
            shown_conf = conf

        # TTS (speak only when new & accepted & cooldown passed)
        now = time.time()
        if shown_label != "Detecting..." and shown_label != last_spoken and (now - last_say_time) > SAY_COOLDOWN:
            to_say = SAY_MAP.get(shown_label, shown_label)
            try:
                engine.say(to_say)
                engine.runAndWait()
            except Exception:
                pass
            last_spoken = shown_label
            last_say_time = now

    # FPS calc
    frame_count += 1
    if frame_count >= 10:
        t1 = time.time()
        fps = frame_count / (t1 - t0 + 1e-9)
        t0 = t1
        frame_count = 0

    # UI overlay
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    line1 = f"Label: {shown_label}"
    if shown_label != "Detecting...":
        line1 += f"  ({shown_conf:.2f})"
    cv2.putText(frame, line1, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Real-Time ISL Recognition (classic model)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

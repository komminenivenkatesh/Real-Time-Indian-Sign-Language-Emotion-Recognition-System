import os
import cv2
import numpy as np
import joblib

# optional heavy deps
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    import mediapipe as mp
except Exception:
    mp = None

MODEL_PATH = "data/models/isl_lstm_model.pt"
ENCODER_PATH = "data/models/label_encoder.pkl"
INPUT_DIM = 126  # 2 hands * 21 landmarks * (x,y,z)
HIDDEN = 128
LAYERS = 2
CONF_THRESH = 0.60


# --- Model (same as training) ---
if torch is not None and nn is not None:
    class BiLSTMClassifier(nn.Module):
        def __init__(self, input_dim, hidden, layers, num_classes):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.3
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Dropout(0.3),
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden, num_classes),
            )

        def forward(self, x):  # x: (B,T,D)
            out, _ = self.lstm(x)
            feat = out[:, -1, :]
            return self.head(feat)
else:
    class BiLSTMClassifier:
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch is required to instantiate BiLSTMClassifier — install 'torch'")
    def __init__(self, input_dim, hidden, layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):  # x: (B,T,D)
        out, _ = self.lstm(x)
        feat = out[:, -1, :]
        return self.head(feat)


# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands if mp is not None else None
mp_drawing = mp.solutions.drawing_utils if mp is not None else None


def extract_126(hand_result, image_w, image_h):
    # return [Lhand(21*3), Rhand(21*3)] flattened; zeros when missing
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
        # sort hands: Left first, then Right
        hands = list(zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness))
        # enforce left/right ordering
        L, R = None, None
        for lm, hd in hands:
            label = hd.classification[0].label
            if label.lower().startswith("left") and L is None:
                L = lm
            elif label.lower().startswith("right") and R is None:
                R = lm
        if L:
            for i, lm in enumerate(L.landmark):
                left[i] = [lm.x, lm.y, lm.z]
        if R:
            for i, lm in enumerate(R.landmark):
                right[i] = [lm.x, lm.y, lm.z]
    feat = np.concatenate([left.reshape(-1), right.reshape(-1)], axis=0)  # (126,)
    return feat


def main():
    if torch is None:
        raise ImportError("PyTorch is required to run realtime ISL LSTM — install 'torch' to use this feature")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load encoder and model
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder not found at {ENCODER_PATH}. Please train or provide the file.")
    le = joblib.load(ENCODER_PATH)
    num_classes = len(le.classes_)
    model = BiLSTMClassifier(INPUT_DIM, HIDDEN, LAYERS, num_classes).to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train or provide the file.")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("🎥 Webcam started. Press 'q' to quit.")
    if mp_hands is None:
        raise ImportError("mediapipe is required to run realtime ISL LSTM — install 'mediapipe' to use this feature")

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            # Draw landmarks
            if res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            feat = extract_126(res, frame.shape[1], frame.shape[0])  # (126,)
            x = torch.tensor(feat, dtype=torch.float32).view(1, 1, INPUT_DIM).to(device)  # (B=1,T=1,D=126)

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                idx = int(probs.argmax())
                conf = float(probs[idx])
                pred = le.inverse_transform([idx])[0] if conf >= CONF_THRESH else "…"

            # Show prediction
            text = f"{pred} ({conf:.2f})" if pred != "…" else "Detecting…"
            cv2.rectangle(frame, (10, 10), (330, 60), (0, 0, 0), -1)
            cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("Real-time ISL LSTM", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

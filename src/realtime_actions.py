import os
import cv2
import numpy as np
import joblib
from collections import deque

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

# must match training
SEQ_LEN = 30

LE_PATH = "data/models/action_label_encoder.pkl"
MODEL_PATH = "data/models/action_bilstm.pt"


if torch is not None and nn is not None:
    class BiLSTM(nn.Module):
        def __init__(self, input_dim, hidden, layers, num_cls):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.2
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Dropout(0.3),
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden, num_cls),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            feat = out.mean(1)
            return self.head(feat)
else:
    class BiLSTM:
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch is required to instantiate BiLSTM — install 'torch' to use realtime actions.")


mp_holistic = mp.solutions.holistic if mp is not None else None


def extract_vec(res):
    vec = []
    if res.pose_landmarks:
        for lm in res.pose_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0] * (33 * 3))
    if res.left_hand_landmarks:
        for lm in res.left_hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0] * (21 * 3))
    if res.right_hand_landmarks:
        for lm in res.right_hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0] * (21 * 3))
    return np.array(vec, dtype=np.float32)


D = 33 * 3 + 21 * 3 + 21 * 3


def main():
    if not os.path.exists(LE_PATH):
        raise FileNotFoundError(f"Label encoder not found at {LE_PATH}. Please train or provide the file.")
    le = joblib.load(LE_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train or provide the file.")
    if torch is None:
        raise ImportError("PyTorch is required to run realtime actions — install 'torch' to use this feature")

    model = BiLSTM(D, 128, 2, len(le.classes_))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("⚠️ No webcam found!")
        return

    buffer = deque(maxlen=SEQ_LEN)
    if mp_holistic is None:
        raise ImportError("mediapipe is required to run realtime actions — install 'mediapipe' to use this feature")

    with mp_holistic.Holistic(model_complexity=1) as holistic:
        print("✅ Webcam started. Press 'q' to quit.")
        last = ""
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)

            vec = extract_vec(res)
            buffer.append(vec)

            if res.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if res.left_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if res.right_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            if len(buffer) == SEQ_LEN:
                x = torch.tensor(np.stack(buffer)[None, ...], dtype=torch.float32)  # (1,T,D)
                with torch.no_grad():
                    logits = model(x)
                    idx = logits.argmax(1).item()
                    label = le.inverse_transform([idx])[0]
                if label != last:
                    print("👉", label)
                    last = label
                cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)

            cv2.imshow("Real-time Action Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

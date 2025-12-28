import os
import time
import sys
import numpy as np
import cv2

# optional heavy dependency
try:
    import mediapipe as mp
except Exception:
    mp = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from joblib import dump

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REC_DIR = os.path.join(DATA_DIR, "recordings")
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(REC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

mp_hands = mp.solutions.hands if mp is not None else None
mp_drawing = mp.solutions.drawing_utils if mp is not None else None


def extract_hand_landmarks(results):
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            label = handedness.classification[0].label
            if label == "Left":
                left = coords
            else:
                right = coords
    # normalize
    left -= left[0]
    right -= right[0]
    feat = np.concatenate([left.flatten(), right.flatten()], axis=0)
    return feat


def record_class(label, seconds=12, min_conf=0.5):
    label_dir = os.path.join(REC_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        sys.exit(1)
    print(f"Recording label '{label}' for {seconds}s…")
    frames = []
    with mp_hands.Hands(
        model_complexity=1, max_num_hands=2, min_detection_confidence=min_conf, min_tracking_confidence=min_conf
    ) as hands:
        t0 = time.time()
        while time.time() - t0 < seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for hlms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)
            feat = extract_hand_landmarks(results)
            frames.append(feat)
            cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
    arr = np.stack(frames)
    out_path = os.path.join(label_dir, f"{int(time.time())}.npy")
    np.save(out_path, arr)
    print(f"✅ Saved {arr.shape[0]} frames at {out_path}")


def load_dataset():
    X, y = [], []
    for label in sorted(os.listdir(REC_DIR)):
        ldir = os.path.join(REC_DIR, label)
        if not os.path.isdir(ldir):
            continue
        for f in os.listdir(ldir):
            if f.endswith(".npy"):
                arr = np.load(os.path.join(ldir, f))
                for row in arr:
                    X.append(row)
                    y.append(label)
    if not X:
        print("❌ No data found. Record some samples first.")
        sys.exit(1)
    return np.array(X, dtype=np.float32), np.array(y)


def train_and_save():
    X, y = load_dataset()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    print("=== Report ===")
    print(classification_report(yte, ypred, target_names=le.classes_))
    dump(clf, os.path.join(MODEL_DIR, "landmarks_model.pkl"))
    dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print("✅ Model saved to data/models/")


if __name__ == "__main__":
    print("Options:")
    print("1) Record a label")
    print("2) Train model from recordings")
    choice = input("Choose (1/2): ").strip()
    if choice == "1":
        label = input("Enter label name (e.g., HELLO, THANKYOU): ").strip().upper()
        seconds = input("Seconds to record (default 12): ").strip()
        seconds = int(seconds) if seconds else 12
        record_class(label, seconds)
    elif choice == "2":
        train_and_save()
    else:
        print("Bye.")

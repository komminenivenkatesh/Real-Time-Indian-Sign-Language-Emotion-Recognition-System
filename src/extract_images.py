import os
import time
import glob
import numpy as np
import cv2

# optional heavy dependency
try:
    import mediapipe as mp
except Exception:
    mp = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
INPUT_ROOT = os.path.join(PROJECT_ROOT, "data", "Indian")  # your dataset folders
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data", "recordings")  # where .npy files go

os.makedirs(OUTPUT_ROOT, exist_ok=True)

mp_hands = mp.solutions.hands if mp is not None else None


def extract_from_image(img_bgr, hands):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)

    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    if res.multi_hand_landmarks and res.multi_handedness:
        for hlms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hlms.landmark], dtype=np.float32)
            lbl = handed.classification[0].label  # "Left" or "Right"
            if lbl == "Left":
                left = coords
            else:
                right = coords
    left -= left[0]
    right -= right[0]
    feat = np.concatenate([left.flatten(), right.flatten()], axis=0)  # (126,)
    # accept only if at least one hand was detected
    if np.all(feat == 0):
        return None
    return feat


def main():
    if mp is None:
        raise ImportError("mediapipe is required to run extract_images.py — install it or skip this step")

    total = 0
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        model_complexity=1, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:

        for label in sorted(os.listdir(INPUT_ROOT)):
            label_dir = os.path.join(INPUT_ROOT, label)
            if not os.path.isdir(label_dir):
                continue

            clean_label = label.strip().replace(" ", "").replace("-", "").upper()
            out_dir = os.path.join(OUTPUT_ROOT, clean_label)
            os.makedirs(out_dir, exist_ok=True)

            patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
            files = []
            for p in patterns:
                files.extend(glob.glob(os.path.join(label_dir, p)))
            if not files:
                continue

            print(f"\n📁 {label}  →  {clean_label}  ({len(files)} images)")
            for fp in files:
                img = cv2.imread(fp)
                if img is None:
                    print("  ⚠️ could not read:", fp)
                    continue
                feat = extract_from_image(img, hands)
                if feat is None:
                    # no hands detected; skip
                    continue
                arr = feat.reshape(1, -1).astype(np.float32)  # (1,126)
                base = os.path.splitext(os.path.basename(fp))[0]
                ts = int(time.time() * 1000)
                out_path = os.path.join(out_dir, f"{base}_{ts}.npy")
                np.save(out_path, arr)
                total += 1
                if total % 200 == 0:
                    print(f"  saved {total} samples…")

    print(f"\n✅ Done. Saved {total} landmark files to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

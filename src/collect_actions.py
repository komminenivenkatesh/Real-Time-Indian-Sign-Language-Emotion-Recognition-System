import os
import time
import cv2
import numpy as np

# heavy optional dependency
try:
    import mediapipe as mp
except Exception:
    mp = None

SAVE_DIR = "data/actions"  # will save sequences per label
ACTIONS = ["WAVE", "CLAP", "THUMBSUP"]  # change/add your actions
SEQ_LEN = 30  # frames per sequence
SAMPLES_PER_ACTION = 30  # sequences per action

def extract_vec(results):
    vec = []
    # Pose (33)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0] * (33 * 3))
    # Left hand (21)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0] * (21 * 3))
    # Right hand (21)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0] * (21 * 3))
    return np.array(vec, dtype=np.float32)


def main():
    if mp is None:
        raise ImportError("mediapipe is required to run collect_actions.py — install it or run only non-hardware modules")

    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils

    os.makedirs(SAVE_DIR, exist_ok=True)
    for a in ACTIONS:
        os.makedirs(os.path.join(SAVE_DIR, a), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("⚠️ No webcam found!")
        return

    with mp_holistic.Holistic(model_complexity=1) as holistic:
        for action in ACTIONS:
            for sample_idx in range(SAMPLES_PER_ACTION):
                print(f"\nGet ready for: {action} | Sample {sample_idx+1}/{SAMPLES_PER_ACTION}")
                time.sleep(1.5)
                seq = []
                collected = 0

                while collected < SEQ_LEN:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = holistic.process(rgb)

                    # draw (optional)
                    if res.pose_landmarks:
                        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    if res.left_hand_landmarks:
                        mp_draw.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if res.right_hand_landmarks:
                        mp_draw.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    vec = extract_vec(res)
                    seq.append(vec)
                    collected += 1

                    cv2.putText(
                        frame,
                        f"{action} | frame {collected}/{SEQ_LEN}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Collect Actions", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                seq = np.stack(seq, axis=0)  # (T,D)
                out_path = os.path.join(SAVE_DIR, action, f"{sample_idx:04d}.npy")
                np.save(out_path, seq)
                print(f"💾 Saved {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Finished collecting!")


if __name__ == "__main__":
    main()

import cv2

# optional dependency
try:
    from fer.fer import FER
except Exception:
    FER = None


def main():
    if FER is None:
        raise ImportError("Package 'fer' not installed — run 'pip install fer' to use realtime_face_emotion.py")

    detector = FER(mtcnn=False)  # simple/fast; set True for more accurate but slower
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("⚠️ No webcam found!")
        return

    print("✅ Webcam started. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # FER expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # returns list of dicts with box + emotions
        results = detector.detect_emotions(rgb)

        for r in results:
            (x, y, w, h) = r["box"]
            em = r["emotions"]
            # Top emotion
            label = max(em, key=em.get)
            score = em[label]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({score:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Live Facial Expression", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

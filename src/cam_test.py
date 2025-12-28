import cv2


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("No camera")
        return
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("cam test", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

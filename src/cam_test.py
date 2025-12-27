import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("No camera"); raise SystemExit
while True:
    ok, frame = cap.read()
    if not ok: break
    cv2.imshow("cam test", frame)
    if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
        break
cap.release()
cv2.destroyAllWindows()

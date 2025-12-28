import cv2
import glob
import os

# optional dependency
try:
    from fer.fer import FER
except Exception:
    FER = None


def main(base_folder=None):
    if FER is None:
        raise ImportError("Package 'fer' not installed — run 'pip install fer' to enable FER image tests")

    if base_folder is None:
        base_folder = r"C:\Users\kommi\OneDrive\Desktop\ml_project\data\train\happy"

    # Find the first available image (JPG/PNG)
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(base_folder, ext)))

    if not files:
        raise FileNotFoundError(f"No images found in {base_folder}")

    img_path = files[0]
    print(f"Using image: {img_path}")

    # Read the image
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    # Initialize FER detector
    detector = FER(mtcnn=True)

    # Detect emotions
    results = detector.detect_emotions(image)
    print("Detections:", results)

    if results:
        for i, face in enumerate(results):
            print(f"\nFace {i+1} emotion scores:")
            for k, v in face["emotions"].items():
                print(f"  {k}: {v:.2f}")
            top = max(face["emotions"], key=face["emotions"].get)
            print("Top emotion:", top)
    else:
        print("No face detected in the image.")


if __name__ == "__main__":
    main()

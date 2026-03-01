import cv2
import argparse
from detect import FaceDetector
from enhance import FaceEnhancer
from recognize import FaceRecognizer
from attendance_manager import AttendanceManager
from utils import quality_check

MODEL_ROOT = 'models/insightface_models'
GFPGAN_MODEL = 'models/insightface_models/GFPGANv1.4.pth'
EMBEDDINGS = 'data/embeddings.pkl'
ATTENDANCE_FILE = 'attendance/attendance.xlsx'


def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detector = FaceDetector(MODEL_ROOT)
    recognizer = FaceRecognizer(MODEL_ROOT, EMBEDDINGS)
    enhancer = FaceEnhancer(GFPGAN_MODEL)
    attendance = AttendanceManager(ATTENDANCE_FILE)

    faces = detector.detect(image)
    print(f"Detected {len(faces)} faces")

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_img = image[y1:y2, x1:x2]

        if not quality_check(face_img):
            face_img = enhancer.enhance(face_img)

        embedding = face.normed_embedding
        match = recognizer.recognize(embedding)

        if match:
            print(f"Recognized: {match['name']}")
            attendance.mark(match['student_id'], match['name'])
        else:
            print("Face not recognized.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Face Attendance Pipeline")
    parser.add_argument('--image', required=True, help='Path to input image')
    args = parser.parse_args()
    main(args.image)

import cv2
import numpy as np

def is_blurry(img, thresh=100.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < thresh

def brightness_ok(img, low=50, high=200):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mean_v = np.mean(v)
    return low <= mean_v <= high

def quality_check(face_img):
    if face_img is None or face_img.size == 0:
        return False
    if face_img.shape[0] < 80 or face_img.shape[1] < 80:
        return False
    if is_blurry(face_img):
        return False
    if not brightness_ok(face_img):
        return False
    return True

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

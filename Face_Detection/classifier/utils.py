import cv2
import numpy as np
import os
from skimage.feature import hog
from tensorflow.keras.applications import vgg16, resnet50, mobilenet_v2, efficientnet

# Configuration de la détection faciale (Standard OpenCV)
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# --- LOGIQUE DE PRÉTRAITEMENT IDENTIQUE À PYQT5 ---

def apply_clahe_bgr(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def detect_largest_face(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )
    if len(faces) == 0: return None
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    return faces[0]

def crop_face(image_bgr, margin_ratio=0.20):
    face = detect_largest_face(image_bgr)
    if face is None: return image_bgr
    x, y, w, h = face
    mx, my = int(w * margin_ratio), int(h * margin_ratio)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(image_bgr.shape[1], x + w + mx), min(image_bgr.shape[0], y + h + my)
    return image_bgr[y1:y2, x1:x2]

def preprocess_face_image(image_bgr, target_size):
    """La fonction clé qui unifie le traitement"""
    face = crop_face(image_bgr, margin_ratio=0.20)
    face = apply_clahe_bgr(face)
    face = cv2.resize(face, (target_size, target_size))
    return face

# --- FONCTIONS POUR LES MODÈLES ---

def extract_hog_features_web(image_bgr, img_size=128):
    # Applique Crop + CLAHE + Resize avant HOG
    face = preprocess_face_image(image_bgr, img_size)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray, orientations=12, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys",
        transform_sqrt=True, feature_vector=True,
    )
    return features.reshape(1, -1)

def preprocess_for_dl_web(uploaded_file, model_info):
    uploaded_file.seek(0)
    img_size = model_info.get('img_size', 224)
    preprocess_name = model_info.get('preprocess', 'rescale')
    
    # Lecture
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Prétraitement PyQt5 style
    face = preprocess_face_image(image_bgr, img_size)
    img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype("float32")

    # Mapping exact de votre code
    if any(x in preprocess_name for x in ["vgg16", "vgg"]):
        return vgg16.preprocess_input(np.expand_dims(img, axis=0))
    elif "resnet50" in preprocess_name:
        return resnet50.preprocess_input(np.expand_dims(img, axis=0))
    elif "mobilenet" in preprocess_name:
        return mobilenet_v2.preprocess_input(np.expand_dims(img, axis=0))
    elif "efficientnet" in preprocess_name:
        return efficientnet.preprocess_input(np.expand_dims(img, axis=0))
    else:
        return np.expand_dims(img / 255.0, axis=0)

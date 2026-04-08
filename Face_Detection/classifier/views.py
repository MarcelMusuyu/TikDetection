from django.shortcuts import render

# Create your views here.
import os
import json
import joblib
import numpy as np
import warnings
from pathlib import Path

from django.shortcuts import render
from django.conf import settings
from tensorflow.keras.models import load_model
from sklearn.exceptions import InconsistentVersionWarning

from .forms import ImageUploadForm
from .models import EvaluationMetric

# Ignorer les alertes de version de scikit-learn pour nettoyer la console
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Dans classifier/views.py, ligne 18 environ :
from .utils import (
    preprocess_face_image,      # Nouveau nom
    extract_hog_features_web, 
    preprocess_for_dl_web
)


# ============================================================
# CONFIGURATION DES MODELES
# ============================================================
BASE_RESULTS_DIR = Path("C:/Users/DELL/Desktop/FICHE/results_face_classification")

MODEL_CONFIGS = {
    "CNN Simple": {
        "type": "dl", 
        "path": BASE_RESULTS_DIR / "deep_learning" / "cnn_simple" / "cnn_simple.keras", 
        "config": BASE_RESULTS_DIR / "deep_learning" / "cnn_simple" / "config.json"
    },
    "VGG16": {
        "type": "dl", 
        "path": BASE_RESULTS_DIR / "transfer_learning" / "vgg16" / "vgg16.keras", 
        "config": BASE_RESULTS_DIR / "transfer_learning" / "vgg16" / "config.json"
    },
    "ResNet50": {
        "type": "dl", 
        "path": BASE_RESULTS_DIR / "transfer_learning" / "resnet50" / "resnet50.keras", 
        "config": BASE_RESULTS_DIR / "transfer_learning" / "resnet50" / "config.json"
    },
    "MobileNetV2": {
        "type": "dl", 
        "path": BASE_RESULTS_DIR / "transfer_learning" / "mobilenetv2" / "mobilenetv2.keras", 
        "config": BASE_RESULTS_DIR / "transfer_learning" / "mobilenetv2" / "config.json"
    },
    "EfficientNetB0": {
        "type": "dl", 
        "path": BASE_RESULTS_DIR / "transfer_learning" / "efficientnetb0" / "efficientnetb0.keras", 
        "config": BASE_RESULTS_DIR / "transfer_learning" / "efficientnetb0" / "config.json"
    },
    "KNN HOG": {
        "type": "classical", 
        "path": BASE_RESULTS_DIR / "classical_ml" / "knn_hog" / "model.joblib", 
        "config": BASE_RESULTS_DIR / "classical_ml" / "knn_hog" / "config.json"
    },
    "SVM HOG": {
        "type": "classical", 
        "path": BASE_RESULTS_DIR / "classical_ml" / "svm_hog" / "model.joblib", 
        "config": BASE_RESULTS_DIR / "classical_ml" / "svm_hog" / "config.json"
    },
    "Random Forest HOG": {
        "type": "classical", 
        "path": BASE_RESULTS_DIR / "classical_ml" / "random_forest_hog" / "model.joblib", 
        "config": BASE_RESULTS_DIR / "classical_ml" / "random_forest_hog" / "config.json"
    },
    "Logistic Regression HOG": {
        "type": "classical", 
        "path": BASE_RESULTS_DIR / "classical_ml" / "logistic_regression_hog" / "model.joblib", 
        "config": BASE_RESULTS_DIR / "classical_ml" / "logistic_regression_hog" / "config.json"
    },
}

# ============================================================
# VUES
# ============================================================

import base64 # Import nécessaire pour l'affichage de l'image
# ... vos autres imports

def predict_view(request):
    result = None
    image_base64 = None
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            name = form.cleaned_data['model_choice']
            cfg = MODEL_CONFIGS.get(name)
            img_file =  request.FILES['image'] 
            
            try:
                # 1. Encodage pour affichage Web (Avant traitement)
                
                image_base64 = f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
                
                # 2. Charger la config JSON
                with open(cfg['config'], 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                
                # 3. Prétraitement et Prédiction
                if cfg['type'] == 'classical':
                    # On récupère l'image brute pour que extract_hog_features_web fasse le Crop/CLAHE
                    img_file.seek(0)
                    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    target_sz = model_info.get('img_size', 128)
                    features = extract_hog_features_web(img_bgr, img_size=target_sz)
                    
                    model = joblib.load(cfg['path'])
                    prediction = model.predict(features)[0]
                else:
                    # Deep Learning / Transfer Learning
                    model = load_model(str(cfg['path']))
                    img_data = preprocess_for_dl_web(img_file, model_info)
                    preds = model.predict(img_data)
                    class_idx = np.argmax(preds)
                    acc = model_info.get('test_accuracy', model_info.get('accuracy', 0.0))
                    
                    idx_to_class = model_info.get('idx_to_class', {})
                    prediction = idx_to_class.get(str(class_idx), f"Classe {class_idx}")

                # 4. Sauvegarde en BDD SQLite
                EvaluationMetric.objects.create(
                    model_name=name,
                    accuracy=acc,
                    report_file_path=str(BASE_RESULTS_DIR / "final_report.md")
                )
                
                result = f"Modèle : {name} | Prédiction : {prediction}| Acc. attendue : {acc * 100:.2f}%"

            except Exception as e:
                result = f"Erreur système : {str(e)}"
    else:
        form = ImageUploadForm()
        
    return render(request, 'classifier/predict.html', {
        'form': form, 
        'result': result, 
        'image_base64': image_base64 # On passe l'image au template
    })


def dashboard_view(request):
    history = EvaluationMetric.objects.all().order_by('-evaluation_date')
    
    report_content = ""
    report_path = BASE_RESULTS_DIR / "final_report.md"
    
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()
        except Exception as e:
            report_content = f"Impossible de lire le rapport : {e}"

    return render(request, 'classifier/dashboard.html', {
        'history': history,
        'report_content': report_content
    })

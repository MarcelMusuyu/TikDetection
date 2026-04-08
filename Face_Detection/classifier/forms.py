from django import forms

class ImageUploadForm(forms.Form):
    MODEL_CHOICES = [
        # Deep Learning & Transfer Learning
        ('CNN Simple', 'CNN Simple'),
        ('VGG16', 'VGG16'),
        ('ResNet50', 'ResNet50'),
        ('MobileNetV2', 'MobileNetV2'),
        ('EfficientNetB0', 'EfficientNetB0'),
        
        # Classical ML (HOG)
        ('KNN HOG', 'KNN HOG'),
        ('SVM HOG', 'SVM HOG'),
        ('Random Forest HOG', 'Random Forest HOG'),
        ('Logistic Regression HOG', 'Logistic Regression HOG'),
    ]
    
    image = forms.ImageField(
        label="Photo du visage",
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )
    
    model_choice = forms.ChoiceField(
        choices=MODEL_CHOICES, 
        label="Choisir l'algorithme",
        widget=forms.Select(attrs={'class': 'form-select'})
    )

 # Optionnel : Ajouter un champ texte pour des notes sur le test
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 2, 'placeholder': 'Ex: Test avec faible luminosité'})
    )
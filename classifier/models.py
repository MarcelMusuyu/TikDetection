from django.db import models

# Create your models here.
from django.db import models

class EvaluationMetric(models.Model):
    # On enlève unique=True pour permettre plusieurs tests du même modèle
    model_name = models.CharField(max_length=100) 
    
    # On ajoute une date automatique pour savoir QUAND le test a été fait
    evaluation_date = models.DateTimeField(auto_now_add=True)
    
    # Métriques
    accuracy = models.FloatField()
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    
    # On ajoute un champ pour noter ce qui a changé (ex: "lr=0.001", "batch=32")
    notes = models.TextField(null=True, blank=True, help_text="Notes sur la configuration de ce test")
    
    report_file_path = models.CharField(max_length=255)

    def __str__(self):
        # Affiche le nom et la date pour les différencier dans l'admin
        return f"{self.model_name} ({self.evaluation_date.strftime('%d/%m/%Y %H:%M')})"

    class Meta:
        # On trie par date : le plus récent en premier
        ordering = ['-evaluation_date']

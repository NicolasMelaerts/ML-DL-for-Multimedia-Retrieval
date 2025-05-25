# -*- coding: utf-8 -*-
"""
Fonctions pour le calcul et l'affichage des métriques d'évaluation
"""

import os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MetricsWindow(QtWidgets.QDialog):
    def __init__(self, parent=None, metrics_data=None):
        super(MetricsWindow, self).__init__(parent)
        self.setWindowTitle("Métriques d'évaluation")
        self.resize(600, 600)
        self.metrics_data = metrics_data or {}
        self.setupUi()
        
    def setupUi(self):
        # Layout principal
        self.mainLayout = QtWidgets.QVBoxLayout(self)
        
        # Information sur les descripteurs combinés
        if hasattr(self.parent(), 'selected_descriptors') and len(self.parent().selected_descriptors) > 1:
            info_label = QtWidgets.QLabel(f"Métriques calculées avec combinaison de descripteurs: {', '.join(self.parent().selected_descriptors)}")
            info_label.setStyleSheet("font-weight: bold; color: blue;")
            self.mainLayout.addWidget(info_label)
        
        # Tableau des métriques
        self.metricsTable = QtWidgets.QTableWidget(5, 2)
        self.metricsTable.setHorizontalHeaderLabels(["Métrique", "Valeur"])
        self.metricsTable.setVerticalHeaderLabels(["", "", "", "", ""])
        
        # Augmenter la hauteur des lignes pour voir toutes les données
        self.metricsTable.verticalHeader().setDefaultSectionSize(30)
        
        # Définir une largeur minimale pour la colonne des valeurs
        self.metricsTable.setColumnWidth(1, 150)
        
        # Définir une hauteur minimale pour le tableau
        self.metricsTable.setMinimumHeight(180)
        
        # Remplir le tableau avec les noms des métriques
        metrics = ["Rappel", "Précision", "AP", "MAP", "R-Precision"]
        for i, metric in enumerate(metrics):
            item = QtWidgets.QTableWidgetItem(metric)
            self.metricsTable.setItem(i, 0, item)
            
            # Initialiser les valeurs à N/A ou aux valeurs fournies
            value = self.metrics_data.get(metric, "N/A")
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            value_item = QtWidgets.QTableWidgetItem(value_str)
            self.metricsTable.setItem(i, 1, value_item)
        
        # Étirer les colonnes pour remplir l'espace disponible
        self.metricsTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.mainLayout.addWidget(self.metricsTable)
        
        # Courbe Rappel/Précision
        self.rpFigure = Figure(figsize=(6, 5), dpi=100)
        self.rpCanvas = FigureCanvas(self.rpFigure)
        self.mainLayout.addWidget(self.rpCanvas)
        
        # Si des données de courbe sont disponibles, les afficher
        if 'precision_recall_curve' in self.metrics_data:
            self.plot_precision_recall_curve(self.metrics_data['precision_recall_curve'])
        
        # Bouton de fermeture
        self.closeButton = QtWidgets.QPushButton("Fermer")
        self.closeButton.clicked.connect(self.accept)
        self.mainLayout.addWidget(self.closeButton)
    
    def plot_precision_recall_curve(self, data):
        """Affiche la courbe précision-rappel"""
        try:
            ax = self.rpFigure.add_subplot(111)
            ax.clear()
            
            # Extraire les données
            recall = data.get('recall', [])
            precision = data.get('precision', [])
            
            if recall and precision and len(recall) == len(precision):
                ax.plot(recall, precision, 'b-', linewidth=2)
                ax.set_xlabel('Rappel')
                ax.set_ylabel('Précision')
                ax.set_title('Courbe Précision-Rappel')
                ax.grid(True)
                ax.set_xlim([0.0, max(0.1, max(recall))])
                ax.set_ylim([0.0, 1.05])
                self.rpCanvas.draw()
        except Exception as e:
            print(f"Erreur lors de l'affichage de la courbe: {str(e)}")

def calculate_metrics(results, image_path, class_counts):
    """
    Calcule les métriques d'évaluation pour une recherche d'images
    
    Args:
        results: Liste des résultats [(path, desc, dist), ...]
        image_path: Chemin de l'image requête
        class_counts: Dictionnaire avec le nombre d'images par classe
        
    Returns:
        Un dictionnaire contenant les métriques calculées
    """
    metrics_data = {}
    
    # Déterminer la classe de l'image requête
    parts = image_path.split(os.sep)
    if len(parts) >= 3:
        # Format attendu: .../animal/race/image.jpg
        animal_idx = max(0, len(parts) - 3)
        breed_idx = max(0, len(parts) - 2)
        req_class = f"{parts[animal_idx]}/{parts[breed_idx]}"
    else:
        print(f"Impossible de déterminer la classe de l'image requête: {image_path}")
        return metrics_data
    
    print(f"Classe de l'image requête: {req_class}")
    
    # Calculer les métriques
    relevant_count = class_counts.get(req_class, 0)
    if relevant_count == 0:
        print(f"Aucune image trouvée pour la classe {req_class}")
        return metrics_data
    
    # Initialiser les listes pour les calculs
    relevants = []
    precisions = []
    recalls = []
    
    # Calculer précision et rappel à chaque rang
    retrieved_relevant = 0
    for i, (path, _, _) in enumerate(results):
        parts = path.split(os.sep)
        if len(parts) >= 3:
            # Format attendu: .../animal/race/image.jpg
            animal_idx = max(0, len(parts) - 3)
            breed_idx = max(0, len(parts) - 2)
            result_class = f"{parts[animal_idx]}/{parts[breed_idx]}"
            
            # Vérifier si le résultat est pertinent (même classe)
            is_relevant = (result_class == req_class)
            relevants.append(is_relevant)
            
            if is_relevant:
                retrieved_relevant += 1
            
            # Calculer précision et rappel à ce rang
            precision = retrieved_relevant / (i + 1)
            recall = retrieved_relevant / relevant_count
            
            precisions.append(precision)
            recalls.append(recall)
    
    # Calculer la précision moyenne (AP)
    ap = 0.0
    if recalls:
        # Utiliser la méthode de l'interpolation
        for i in range(11):  # 11 points: 0.0, 0.1, ..., 1.0
            r = i / 10
            # Trouver toutes les précisions à des rappels >= r
            p_at_r = [precisions[j] for j in range(len(recalls)) if recalls[j] >= r]
            if p_at_r:
                ap += max(p_at_r) / 11
    
    # Calculer R-Precision
    r_precision = 0.0
    if relevant_count <= len(relevants):
        r_precision = sum(relevants[:relevant_count]) / relevant_count
    
    # Stocker les métriques dans le dictionnaire
    metrics_data = {
        "Rappel": recalls[-1] if recalls else 0.0,
        "Précision": precisions[-1] if precisions else 0.0,
        "AP": ap,
        "MAP": ap,  # Pour une seule requête, MAP = AP
        "R-Precision": r_precision,
        "precision_recall_curve": {
            "recall": recalls,
            "precision": precisions
        }
    }
    
    print(f"Métriques calculées: Rappel={metrics_data['Rappel']:.4f}, "
          f"Précision={metrics_data['Précision']:.4f}, AP={metrics_data['AP']:.4f}, "
          f"R-Precision={metrics_data['R-Precision']:.4f}")
    
    return metrics_data 
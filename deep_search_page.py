# -*- coding: utf-8 -*-
"""
Page de moteur de recherche d'images par deep learning
"""

from PyQt5 import QtCore, QtGui, QtWidgets
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
from distances import getkVoisins_deep

class DeepSearchPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DeepSearchPage, self).__init__(parent)
        self.setupUi()
        self.filenames = "MIR_DATASETS_B"  # Dossier contenant la base d'images
        self.features_dict = {}  # Dictionnaire pour stocker les descripteurs par modèle
        self.image_dict = {}  # Dictionnaire pour stocker les chemins des images
        self.model_choice = 0
        self.image_path = ""
        self.query_name = ""  # Nom de l'image requête (sans extension)
        self.results = []
        self.class_counts = {}  # Pour stocker le nombre d'images par classe
        self.metrics_data = {}  # Pour stocker les métriques d'évaluation
        
    def setupUi(self):
        self.setObjectName("DeepSearchPage")
        self.resize(1200, 800)
        
        # Layout principal
        self.mainLayout = QtWidgets.QVBoxLayout(self)
        
        # Titre
        self.titleLabel = QtWidgets.QLabel("Moteur de Recherche d'Images par Deep Learning")
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.titleLabel.setFont(font)
        self.mainLayout.addWidget(self.titleLabel)
        
        # Layout horizontal pour la partie supérieure (3 zones)
        self.topLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.addLayout(self.topLayout)
        
        # ZONE 1: Panneau de contrôle (modèles, affichage)
        self.controlPanel = QtWidgets.QGroupBox("Contrôles")
        self.controlLayout = QtWidgets.QVBoxLayout(self.controlPanel)
        self.topLayout.addWidget(self.controlPanel)
        
        # Sélection des modèles
        self.modelsGroup = QtWidgets.QGroupBox("Modèles Deep Learning")
        self.modelsLayout = QtWidgets.QGridLayout(self.modelsGroup)
        self.controlLayout.addWidget(self.modelsGroup)
        
        # Checkboxes pour les modèles
        self.checkBoxGoogLeNet = QtWidgets.QCheckBox("GoogLeNet")
        self.modelsLayout.addWidget(self.checkBoxGoogLeNet, 0, 0)
        
        self.checkBoxInception = QtWidgets.QCheckBox("Inception v3")
        self.modelsLayout.addWidget(self.checkBoxInception, 0, 1)
        
        self.checkBoxResNet = QtWidgets.QCheckBox("ResNet")
        self.modelsLayout.addWidget(self.checkBoxResNet, 1, 0)
        
        self.checkBoxViT = QtWidgets.QCheckBox("ViT")
        self.modelsLayout.addWidget(self.checkBoxViT, 1, 1)
        
        # Ajouter VGG
        self.checkBoxVGG = QtWidgets.QCheckBox("VGG")
        self.modelsLayout.addWidget(self.checkBoxVGG, 2, 0)
        
        # Sélection du nombre de résultats
        self.displayGroup = QtWidgets.QGroupBox("Affichage")
        self.displayLayout = QtWidgets.QVBoxLayout(self.displayGroup)
        self.controlLayout.addWidget(self.displayGroup)
        
        self.displayComboBox = QtWidgets.QComboBox()
        self.displayComboBox.addItems(["Top 20", "Top 50"])
        self.displayLayout.addWidget(self.displayComboBox)
        
        # ZONE 2: Boutons d'action
        self.buttonPanel = QtWidgets.QGroupBox("Actions")
        self.buttonLayout = QtWidgets.QVBoxLayout(self.buttonPanel)
        self.topLayout.addWidget(self.buttonPanel)
        
        self.loadFeaturesButton = QtWidgets.QPushButton("Charger les features")
        self.loadFeaturesButton.setMinimumHeight(40)
        self.buttonLayout.addWidget(self.loadFeaturesButton)
        
        self.loadImageButton = QtWidgets.QPushButton("Charger une image")
        self.loadImageButton.setMinimumHeight(40)
        self.buttonLayout.addWidget(self.loadImageButton)
        
        self.searchButton = QtWidgets.QPushButton("Rechercher")
        self.searchButton.setMinimumHeight(40)
        self.buttonLayout.addWidget(self.searchButton)
        
        # Ajouter le bouton des métriques ici
        self.metricsButton = QtWidgets.QPushButton("Voir les métriques")
        self.metricsButton.setMinimumHeight(40)
        self.metricsButton.setEnabled(False)  # Désactivé par défaut
        self.buttonLayout.addWidget(self.metricsButton)
        
        # Ajouter un espace extensible pour pousser les boutons vers le haut
        self.buttonLayout.addStretch()
        
        # ZONE 3: Panneau d'image requête
        self.requestPanel = QtWidgets.QGroupBox("Image Requête")
        self.requestLayout = QtWidgets.QVBoxLayout(self.requestPanel)
        self.topLayout.addWidget(self.requestPanel)
        
        # Image requête
        self.requestImageLabel = QtWidgets.QLabel()
        self.requestImageLabel.setFixedSize(250, 250)
        self.requestImageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.requestImageLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.requestImageLabel.setText("Aucune image chargée")
        self.requestLayout.addWidget(self.requestImageLabel, 0, QtCore.Qt.AlignCenter)
        
        # Layout horizontal pour la partie inférieure
        self.bottomLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.addLayout(self.bottomLayout)
        
        # Panneau de résultats (occupe tout l'espace)
        self.resultsPanel = QtWidgets.QGroupBox("Résultats de la Recherche")
        self.resultsLayout = QtWidgets.QVBoxLayout(self.resultsPanel)
        self.bottomLayout.addWidget(self.resultsPanel)
        
        # Scroll area pour les résultats
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.resultsLayout.addWidget(self.scrollArea)
        
        self.scrollContent = QtWidgets.QWidget()
        self.scrollArea.setWidget(self.scrollContent)
        
        self.resultsGrid = QtWidgets.QGridLayout(self.scrollContent)
        
        # Barre de progression
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setValue(0)
        self.mainLayout.addWidget(self.progressBar)
        
        # Bouton Retour en bas de la page
        self.backButton = QtWidgets.QPushButton("Retour à l'accueil")
        self.backButton.setMinimumHeight(40)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.backButton.setFont(font)
        self.mainLayout.addWidget(self.backButton)
        
        # Connecter les signaux
        self.connectSignals()
    
    def connectSignals(self):
        """Connecte les signaux aux slots"""
        self.loadFeaturesButton.clicked.connect(self.loadFeatures)
        self.loadImageButton.clicked.connect(self.loadImage)
        self.searchButton.clicked.connect(self.search)
        self.metricsButton.clicked.connect(self.showMetricsWindow)
    
    def loadImage(self):
        """Charge une image requête"""
        self.image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Sélectionner une image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if self.image_path:
            # Extraire le nom de base de l'image (sans extension)
            self.query_name = os.path.splitext(os.path.basename(self.image_path))[0]
            
            # Charger l'image avec OpenCV pour pouvoir la redimensionner
            img = cv2.imread(self.image_path)
            if img is not None:
                # Redimensionner l'image à une taille plus petite (par exemple 300x300 max)
                height, width = img.shape[:2]
                max_size = 200
                if height > max_size or width > max_size:
                    # Calculer le ratio pour préserver les proportions
                    ratio = min(max_size / width, max_size / height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    img = cv2.resize(img, (new_width, new_height))
                
                # Convertir BGR en RGB pour l'affichage
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Créer un QPixmap à partir de l'image
                height, width, channel = img.shape
                bytesPerLine = 3 * width
                qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                
                # Afficher l'image
                self.requestImageLabel.setPixmap(pixmap)
                self.requestImageLabel.setScaledContents(True)
                
                print(f"Image chargée: {self.image_path}")
                print(f"Nom de l'image requête: {self.query_name}")
                
                # Ajouter l'image au dictionnaire d'images
                self.image_dict[self.query_name] = self.image_path
            else:
                QtWidgets.QMessageBox.warning(self, "Erreur", 
                                             "Impossible de charger l'image sélectionnée.")
    
    def loadFeatures(self):
        """Charge les features des modèles deep learning"""
        # Réinitialiser les features
        self.features_dict = {}
        self.image_dict = {}
        
        # Réinitialiser la barre de progression
        self.progressBar.setValue(0)
        
        # Vérifier quels modèles sont sélectionnés
        selected_models = []
        if self.checkBoxGoogLeNet.isChecked():
            selected_models.append("GoogLeNet")
        if self.checkBoxInception.isChecked():
            selected_models.append("Inception")
        if self.checkBoxResNet.isChecked():
            selected_models.append("ResNet")
        if self.checkBoxViT.isChecked():
            selected_models.append("ViT")
        if self.checkBoxVGG.isChecked():
            selected_models.append("VGG")
        
        if not selected_models:
            QtWidgets.QMessageBox.warning(self, "Attention", 
                                         "Veuillez sélectionner au moins un modèle.")
            return
        
        # Charger les features pour chaque modèle sélectionné
        total_models = len(selected_models)
        
        # Désactiver les boutons pendant le chargement
        self.loadFeaturesButton.setEnabled(False)
        self.loadImageButton.setEnabled(False)
        self.searchButton.setEnabled(False)
        
        try:
            for i, model_name in enumerate(selected_models):
                # Afficher le modèle en cours de chargement
                self.progressBar.setFormat(f"Préparation du modèle {model_name}: %p%")
                self.progressBar.setValue(0)
                QtWidgets.QApplication.processEvents()
                
                # Charger les features du modèle
                features_dict, image_dict = self.load_features_with_images(
                    f"/opt/TP/Features/{model_name}", self.filenames)
                
                if features_dict:
                    self.features_dict[model_name] = features_dict
                    # Fusionner les dictionnaires d'images
                    self.image_dict.update(image_dict)
                    print(f"Chargé {len(features_dict)} features pour {model_name}")
                else:
                    print(f"Aucune feature chargée pour {model_name}")
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Erreur", 
                                          f"Une erreur s'est produite lors du chargement des features: {str(e)}")
        
        finally:
            # Réactiver les boutons
            self.loadFeaturesButton.setEnabled(True)
            self.loadImageButton.setEnabled(True)
            self.searchButton.setEnabled(True)
            
            # Finaliser la barre de progression
            self.progressBar.setValue(100)
            self.progressBar.setFormat("%p%")  # Remettre le format par défaut
            
            # Afficher un message de succès
            if self.features_dict:
                total_features = sum(len(features) for features in self.features_dict.values())
                QtWidgets.QMessageBox.information(self, "Succès", 
                                                f"Chargé {total_features} features pour {len(self.features_dict)} modèles.")
            else:
                QtWidgets.QMessageBox.warning(self, "Attention", 
                                             "Aucune feature n'a pu être chargée.")
    
    def load_features_with_images(self, feature_folder, image_folder):
        """
        Charge les features et trouve les images correspondantes.
        
        Args:
            feature_folder: Dossier contenant les fichiers de features
            image_folder: Dossier contenant les images
            
        Returns:
            Tuple (features_dict, image_dict) contenant les features et les chemins des images
        """
        feature_files = sorted(glob.glob(os.path.join(feature_folder, "*.txt")))
        features_dict = {}
        image_dict = {}
        
        total_files = len(feature_files)
        print(f"Chargement de {total_files} fichiers depuis {feature_folder}")
        
        # Initialiser la barre de progression pour ce modèle
        self.progressBar.setFormat(f"Chargement de {os.path.basename(feature_folder)}: %p%")
        
        for i, file in enumerate(feature_files):
            try:
                # Mettre à jour la barre de progression
                progress = int(100 * i / total_files)
                self.progressBar.setValue(progress)
                QtWidgets.QApplication.processEvents()  # Forcer la mise à jour de l'interface
                
                feature_vector = np.loadtxt(file, ndmin=1)
                base_name = os.path.splitext(os.path.basename(file))[0]
                
                # Stocker les features
                features_dict[base_name] = feature_vector
                
                # Trouver l'image correspondante
                # D'abord, essayer de trouver l'image directement
                image_path_jpg = os.path.join(image_folder, base_name + ".jpg")
                image_path_jpeg = os.path.join(image_folder, base_name + ".jpeg")
                image_path_png = os.path.join(image_folder, base_name + ".png")
                
                # Vérifier l'existence des images
                if os.path.exists(image_path_jpg):
                    image_dict[base_name] = image_path_jpg
                elif os.path.exists(image_path_jpeg):
                    image_dict[base_name] = image_path_jpeg
                elif os.path.exists(image_path_png):
                    image_dict[base_name] = image_path_png
                else:
                    # Si l'image n'est pas trouvée directement, essayer de la chercher dans les sous-dossiers
                    found_path = find_image_in_directory(image_folder, base_name)
                    if found_path:
                        image_dict[base_name] = found_path
                    else:
                        print(f"Aucune image trouvée pour {file} !")
            except Exception as e:
                print(f"Erreur lors du chargement de {file}: {str(e)}")
        
        # Finaliser la barre de progression pour ce modèle
        self.progressBar.setValue(100)
        self.progressBar.setFormat("%p%")  # Remettre le format par défaut
        
        print(f"{len(features_dict)} caractéristiques chargées avec {len(image_dict)} images depuis {feature_folder}")
        
        # Compter les images par classe pour les métriques
        self.class_counts = self.count_images_by_class(image_dict)
        
        return features_dict, image_dict
    
    def count_images_by_class(self, image_dict):
        """Compte le nombre d'images par classe"""
        class_counts = {}
        for image_path in image_dict.values():
            parts = image_path.split(os.sep)
            if len(parts) >= 3:
                # Format attendu: .../animal/race/image.jpg
                animal_idx = max(0, len(parts) - 3)
                breed_idx = max(0, len(parts) - 2)
                class_key = f"{parts[animal_idx]}/{parts[breed_idx]}"
                class_counts[class_key] = class_counts.get(class_key, 0) + 1
        return class_counts
    
    def search(self):
        """Effectue la recherche d'images similaires"""
        # Vérifier qu'une image est chargée
        if not self.query_name or not self.image_path:
            QtWidgets.QMessageBox.warning(self, "Attention", 
                                         "Veuillez d'abord charger une image requête.")
            return
        
        # Vérifier que des features sont chargées
        if not self.features_dict:
            QtWidgets.QMessageBox.warning(self, "Attention", 
                                         "Veuillez d'abord charger les features.")
            return
        
        # Réinitialiser la barre de progression et les résultats
        self.progressBar.setValue(0)
        self.results = []
        
        # Réinitialiser les métriques au début de la recherche
        self.metrics_data = {}
        self.metricsButton.setEnabled(False)
        
        # Déterminer le nombre de résultats à afficher
        display_choice = self.displayComboBox.currentText()
        if display_choice == "Top 20":
            k = 20
        else:  # "Top 50"
            k = 50
        
        # Effectuer la recherche pour chaque modèle sélectionné
        total_models = len(self.features_dict)
        for i, (model_name, features_dict) in enumerate(self.features_dict.items()):
            # Mettre à jour la barre de progression
            progress = int((i / total_models) * 50)  # 50% pour la recherche
            self.progressBar.setValue(progress)
            QtWidgets.QApplication.processEvents()
            
            # Vérifier si l'image requête a des features pour ce modèle
            if self.query_name in features_dict:
                # Rechercher les k plus proches voisins
                neighbors = getkVoisins_deep(features_dict, self.query_name, k)
                
                # Ajouter les résultats à la liste globale
                for neighbor_name, dist in neighbors:
                    if neighbor_name in self.image_dict:
                        self.results.append((self.image_dict[neighbor_name], dist, model_name))
            else:
                print(f"L'image requête n'a pas de features pour le modèle {model_name}")
        
        # Trier les résultats par distance (croissante)
        self.results.sort(key=lambda x: x[1])
        
        # Limiter aux k premiers résultats
        self.results = self.results[:k]
        
        # Effacer les résultats précédents
        self.clearResults()
        
        # Afficher les nouveaux résultats
        self.displayResults()
        
        # Finaliser la barre de progression
        self.progressBar.setValue(100)
        
        print(f"Recherche terminée: {len(self.results)} résultats affichés")
    
    def clearResults(self):
        """Efface les résultats précédents"""
        # Supprimer tous les widgets du layout des résultats
        while self.resultsGrid.count():
            item = self.resultsGrid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def displayResults(self):
        """Affiche les résultats de la recherche"""
        # Déterminer le nombre de colonnes (5 par défaut)
        cols = 5
        
        # Afficher les images
        for i, (path, dist, model_name) in enumerate(self.results):
            try:
                # Charger l'image
                img = cv2.imread(path)
                if img is None:
                    continue
                
                # Convertir BGR en RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Créer un QPixmap à partir de l'image
                height, width, channel = img.shape
                bytesPerLine = 3 * width
                qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                
                # Créer un widget pour contenir l'image et les informations
                container = QtWidgets.QWidget()
                layout = QtWidgets.QVBoxLayout(container)
                
                # Label pour l'image
                imageLabel = QtWidgets.QLabel()
                imageLabel.setPixmap(pixmap.scaled(150, 150, QtCore.Qt.KeepAspectRatio, 
                                                 QtCore.Qt.SmoothTransformation))
                imageLabel.setAlignment(QtCore.Qt.AlignCenter)
                layout.addWidget(imageLabel)
                
                # Label pour les informations
                infoLabel = QtWidgets.QLabel(f"{os.path.basename(path)}\nDist: {dist:.4f}\nModèle: {model_name}")
                infoLabel.setAlignment(QtCore.Qt.AlignCenter)
                layout.addWidget(infoLabel)
                
                # Ajouter le container au layout des résultats
                row = i // cols
                col = i % cols
                self.resultsGrid.addWidget(container, row, col)
                
            except Exception as e:
                print(f"Erreur lors de l'affichage de l'image {path}: {str(e)}")
        
        # Calculer et stocker les métriques
        self.calculateMetrics()
        
        # Activer le bouton des métriques
        self.metricsButton.setEnabled(True)
    
    def calculateMetrics(self):
        """Calcule les métriques d'évaluation"""
        if not self.results:
            self.metrics_data = {}
            return
        
        # Extraire la classe de l'image requête
        req_class = None
        if self.image_path:
            parts = self.image_path.split(os.sep)
            if len(parts) >= 3:
                # Format attendu: .../animal/race/image.jpg
                animal_idx = max(0, len(parts) - 3)
                breed_idx = max(0, len(parts) - 2)
                req_class = f"{parts[animal_idx]}/{parts[breed_idx]}"
        
        if not req_class:
            print("Impossible de déterminer la classe de l'image requête")
            self.metrics_data = {}
            return
        
        # Calculer les métriques
        relevant_count = self.class_counts.get(req_class, 0)
        if relevant_count == 0:
            print(f"Aucune image trouvée pour la classe {req_class}")
            self.metrics_data = {}
            return
        
        # Initialiser les listes pour les calculs
        relevants = []
        precisions = []
        recalls = []
        
        # Calculer précision et rappel à chaque rang
        retrieved_relevant = 0
        for i, (path, _, _) in enumerate(self.results):
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
        self.metrics_data = {
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
        
        print(f"Métriques calculées: Rappel={self.metrics_data['Rappel']:.4f}, "
              f"Précision={self.metrics_data['Précision']:.4f}, AP={self.metrics_data['AP']:.4f}, "
              f"R-Precision={self.metrics_data['R-Precision']:.4f}")
    
    def showMetricsWindow(self):
        """Affiche la fenêtre des métriques d'évaluation"""
        from search_page import MetricsWindow  # Importer la classe MetricsWindow
        metrics_window = MetricsWindow(self, self.metrics_data)
        metrics_window.exec_()

# Fonction pour trouver une image dans un dossier et ses sous-dossiers
def find_image_in_directory(base_dir, image_name):
    """
    Recherche récursivement une image dans un dossier et ses sous-dossiers.
    
    Args:
        base_dir: Dossier de base pour la recherche
        image_name: Nom de l'image à rechercher (sans extension)
        
    Returns:
        Chemin complet de l'image si trouvée, None sinon
    """
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_name_without_ext = os.path.splitext(file)[0]
                if file_name_without_ext == image_name:
                    return os.path.join(root, file)
    return None 
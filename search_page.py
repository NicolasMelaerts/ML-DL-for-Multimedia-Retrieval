# -*- coding: utf-8 -*-
"""
Page de moteur de recherche d'images
"""

from PyQt5 import QtCore, QtGui, QtWidgets
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
from descriptors import (
    showDialog, 
    extractReqFeatures
)

from distances import getkVoisins

class SearchPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(SearchPage, self).__init__(parent)
        self.setupUi()
        self.filenames = "MIR_DATASETS_B"  # Dossier contenant la base d'images
        self.features = {}  # Dictionnaire pour stocker les descripteurs par type
        self.algo_choice = 0
        self.image_path = ""
        self.results = []
        self.class_counts = {}  # Pour stocker le nombre d'images par classe
        
    def setupUi(self):
        self.setObjectName("SearchPage")
        self.resize(1200, 800)
        
        # Layout principal
        self.mainLayout = QtWidgets.QVBoxLayout(self)
        
        # Titre
        self.titleLabel = QtWidgets.QLabel("Moteur de Recherche d'Images")
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.titleLabel.setFont(font)
        self.mainLayout.addWidget(self.titleLabel)
        
        # Layout horizontal pour la partie supérieure (3 zones)
        self.topLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.addLayout(self.topLayout)
        
        # ZONE 1: Panneau de contrôle (descripteurs, distance, affichage)
        self.controlPanel = QtWidgets.QGroupBox("Contrôles")
        self.controlLayout = QtWidgets.QVBoxLayout(self.controlPanel)
        self.topLayout.addWidget(self.controlPanel)
        
        # Sélection des descripteurs
        self.descriptorsGroup = QtWidgets.QGroupBox("Descripteurs")
        self.descriptorsLayout = QtWidgets.QGridLayout(self.descriptorsGroup)
        self.controlLayout.addWidget(self.descriptorsGroup)
        
        # Checkboxes pour les descripteurs
        self.checkBoxBGR = QtWidgets.QCheckBox("BGR")
        self.descriptorsLayout.addWidget(self.checkBoxBGR, 0, 0)
        
        self.checkBoxHSV = QtWidgets.QCheckBox("HSV")
        self.descriptorsLayout.addWidget(self.checkBoxHSV, 0, 1)
        
        self.checkBoxSIFT = QtWidgets.QCheckBox("SIFT")
        self.descriptorsLayout.addWidget(self.checkBoxSIFT, 1, 0)
        
        self.checkBoxORB = QtWidgets.QCheckBox("ORB")
        self.descriptorsLayout.addWidget(self.checkBoxORB, 1, 1)
        
        self.checkBoxGLCM = QtWidgets.QCheckBox("GLCM")
        self.descriptorsLayout.addWidget(self.checkBoxGLCM, 2, 0)
        
        self.checkBoxLBP = QtWidgets.QCheckBox("LBP")
        self.descriptorsLayout.addWidget(self.checkBoxLBP, 2, 1)
        
        self.checkBoxHOG = QtWidgets.QCheckBox("HOG")
        self.descriptorsLayout.addWidget(self.checkBoxHOG, 3, 0)
        
        # Sélection de la distance
        self.distanceGroup = QtWidgets.QGroupBox("Distance")
        self.distanceLayout = QtWidgets.QVBoxLayout(self.distanceGroup)
        self.controlLayout.addWidget(self.distanceGroup)
        
        self.distanceComboBox = QtWidgets.QComboBox()
        self.distanceComboBox.addItems(["Euclidienne", "Correlation", "Chi carre", "Intersection", "Bhattacharyya"])
        self.distanceLayout.addWidget(self.distanceComboBox)
        
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
        
        self.loadDescriptorsButton = QtWidgets.QPushButton("Charger les descripteurs")
        self.loadDescriptorsButton.setMinimumHeight(40)
        self.buttonLayout.addWidget(self.loadDescriptorsButton)
        
        self.loadImageButton = QtWidgets.QPushButton("Charger une image")
        self.loadImageButton.setMinimumHeight(40)
        self.buttonLayout.addWidget(self.loadImageButton)
        
        self.searchButton = QtWidgets.QPushButton("Rechercher")
        self.searchButton.setMinimumHeight(40)
        self.buttonLayout.addWidget(self.searchButton)
        
        self.backButton = QtWidgets.QPushButton("Retour")
        self.backButton.setMinimumHeight(40)
        self.buttonLayout.addWidget(self.backButton)
        
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
        
        # Panneau de résultats (gauche)
        self.resultsPanel = QtWidgets.QGroupBox("Résultats de la Recherche")
        self.resultsLayout = QtWidgets.QVBoxLayout(self.resultsPanel)
        self.bottomLayout.addWidget(self.resultsPanel, 3)  # Proportion 3
        
        # Scroll area pour les résultats
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.resultsLayout.addWidget(self.scrollArea)
        
        self.scrollContent = QtWidgets.QWidget()
        self.scrollArea.setWidget(self.scrollContent)
        
        self.resultsGrid = QtWidgets.QGridLayout(self.scrollContent)
        
        # Panneau de métriques (droite)
        self.metricsPanel = QtWidgets.QGroupBox("Métriques d'Évaluation")
        self.metricsLayout = QtWidgets.QVBoxLayout(self.metricsPanel)
        self.bottomLayout.addWidget(self.metricsPanel, 2)  # Proportion 2
        
        # Tableau des métriques avec plus d'espace
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
            
            # Initialiser les valeurs à N/A
            value_item = QtWidgets.QTableWidgetItem("N/A")
            self.metricsTable.setItem(i, 1, value_item)
        
        # Étirer les colonnes pour remplir l'espace disponible
        self.metricsTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.metricsLayout.addWidget(self.metricsTable)
        
        # Courbe Rappel/Précision
        self.rpFigure = Figure(figsize=(6, 5), dpi=100)
        self.rpCanvas = FigureCanvas(self.rpFigure)
        self.metricsLayout.addWidget(self.rpCanvas)
        
        # Barre de progression
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setValue(0)
        self.mainLayout.addWidget(self.progressBar)
        
        # Connecter les signaux
        self.connectSignals()
    
    def connectSignals(self):
        """Connecte les signaux aux slots"""
        self.loadDescriptorsButton.clicked.connect(self.loadDescriptors)
        self.loadImageButton.clicked.connect(self.loadImage)
        self.searchButton.clicked.connect(self.search)
        
        # Connecter les changements de descripteurs à la mise à jour des distances
        self.checkBoxBGR.stateChanged.connect(self.updateDistanceOptions)
        self.checkBoxHSV.stateChanged.connect(self.updateDistanceOptions)
        self.checkBoxSIFT.stateChanged.connect(self.updateDistanceOptions)
        self.checkBoxORB.stateChanged.connect(self.updateDistanceOptions)
        self.checkBoxGLCM.stateChanged.connect(self.updateDistanceOptions)
        self.checkBoxLBP.stateChanged.connect(self.updateDistanceOptions)
        self.checkBoxHOG.stateChanged.connect(self.updateDistanceOptions)
    
    def updateDistanceOptions(self):
        """Met à jour les options de distance en fonction des descripteurs sélectionnés"""
        self.distanceComboBox.clear()
        
        # Options de base pour les descripteurs d'histogramme et HOG
        if any([self.checkBoxBGR.isChecked(), self.checkBoxHSV.isChecked(), 
                self.checkBoxGLCM.isChecked(), self.checkBoxLBP.isChecked(),
                self.checkBoxHOG.isChecked()]):
            self.distanceComboBox.addItems(["Euclidienne", "Correlation", "Chi carre", 
                                           "Intersection", "Bhattacharyya"])
        
        # Options pour SIFT
        if self.checkBoxSIFT.isChecked():
            self.distanceComboBox.addItem("Flann")
        
        # Options pour ORB
        if self.checkBoxORB.isChecked():
            self.distanceComboBox.addItem("Brute force")
        
        # Si aucun descripteur n'est sélectionné, ajouter l'option par défaut
        if self.distanceComboBox.count() == 0:
            self.distanceComboBox.addItem("Euclidienne")
    
    def loadImage(self):
        """Charge une image requête"""
        self.image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Sélectionner une image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if self.image_path:
            # Charger l'image avec OpenCV pour pouvoir la redimensionner
            img = cv2.imread(self.image_path)
            if img is not None:
                # Redimensionner l'image à une taille plus petite (par exemple 300x300 max)
                height, width = img.shape[:2]
                max_size = 200
                if height > max_size or width > max_size:
                    # Calculer le ratio pour préserver les proportions
                    ratio = min(max_size / width, max_size / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    img = cv2.resize(img, new_size)
                
                # Convertir l'image OpenCV en QPixmap
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                bytes_per_line = ch * w
                q_img = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_img)
                
                self.requestImageLabel.setPixmap(pixmap)
                self.requestImageLabel.setAlignment(QtCore.Qt.AlignCenter)
            else:
                # Fallback à la méthode originale si OpenCV ne peut pas lire l'image
                pixmap = QtGui.QPixmap(self.image_path)
                pixmap = pixmap.scaled(200, 200, QtCore.Qt.KeepAspectRatio)
                self.requestImageLabel.setPixmap(pixmap)
                self.requestImageLabel.setAlignment(QtCore.Qt.AlignCenter)
    
    def loadFeatureType(self, folder_name, algo_id):
        """Charge un type spécifique de descripteur depuis une structure plate"""
        features = []
        folder_path = f'./{folder_name}'
        
        # Vérifier si le dossier existe
        if not os.path.exists(folder_path):
            print(f"Le dossier {folder_name} n'existe pas.")
            return []  # Retourner une liste vide sans afficher de message
        
        print(f"Chargement des descripteurs depuis {folder_path}")
        
        # Obtenir la liste de tous les fichiers .txt dans le dossier
        all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        total_files = len(all_files)
        
        if total_files == 0:
            print(f"Aucun fichier de descripteur trouvé dans {folder_path}")
            return []
        
        # Traiter les fichiers par lots pour mettre à jour la barre de progression
        batch_size = max(1, total_files // 100)  # Diviser en environ 100 lots
        
        for i in range(0, total_files, batch_size):
            # Traiter un lot de fichiers
            batch_files = all_files[i:i+batch_size]
            
            for file_name in batch_files:
                try:
                    # Charger le descripteur
                    data_path = os.path.join(folder_path, file_name)
                    feature = np.loadtxt(data_path)
                    
                    # Extraire les informations du nom de fichier
                    # Format attendu: animal_race_imagename.txt
                    parts = os.path.splitext(file_name)[0].split('_')
                    
                    if len(parts) >= 3:
                        animal = parts[0]
                        breed = parts[1]
                        image_name = '_'.join(parts[2:]) + '.jpg'
                        
                        # Construire le chemin de l'image
                        image_path = os.path.join(self.filenames, animal, breed, image_name)
                        
                        if os.path.exists(image_path):
                            features.append((image_path, feature))
                        else:
                            print(f"Image introuvable: {image_path}")
                    else:
                        print(f"Format de nom de fichier non reconnu: {file_name}")
                    
                except Exception as e:
                    print(f"Erreur lors du chargement de {file_name}: {str(e)}")
            
            # Mettre à jour la barre de progression
            progress_value = min(100, int(100 * (i + len(batch_files)) / total_files))
            self.progressBar.setValue(progress_value)
            QtWidgets.QApplication.processEvents()  # Forcer la mise à jour de l'interface
        
        print(f"Chargé {len(features)} descripteurs {folder_name}")
        return features
    
    def loadDescriptors(self):
        """Charge les descripteurs sélectionnés"""
        # Vérifier qu'au moins un descripteur est sélectionné
        if not any([
            self.checkBoxBGR.isChecked(),
            self.checkBoxHSV.isChecked(),
            self.checkBoxSIFT.isChecked(),
            self.checkBoxORB.isChecked(),
            self.checkBoxGLCM.isChecked(),
            self.checkBoxLBP.isChecked(),
            self.checkBoxHOG.isChecked()
        ]):
            showDialog()
            return
        
        self.features = {}  # Réinitialiser les descripteurs
        self.progressBar.setValue(0)
        QtWidgets.QApplication.processEvents()  # Forcer la mise à jour de l'interface
        
        # Charger les descripteurs sélectionnés
        total_descriptors = sum([
            self.checkBoxBGR.isChecked(),
            self.checkBoxHSV.isChecked(),
            self.checkBoxSIFT.isChecked(),
            self.checkBoxORB.isChecked(),
            self.checkBoxGLCM.isChecked(),
            self.checkBoxLBP.isChecked(),
            self.checkBoxHOG.isChecked()
        ])
        
        progress = 0
        loaded_descriptors = []
        
        # Charger BGR
        if self.checkBoxBGR.isChecked():
            self.progressBar.setValue(0)  # Réinitialiser pour chaque type de descripteur
            QtWidgets.QApplication.processEvents()
            features_bgr = self.loadFeatureType('BGR', 1)
            if features_bgr:  # Vérifier si des descripteurs ont été chargés
                self.features['BGR'] = features_bgr
                loaded_descriptors.append('BGR')
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger HSV
        if self.checkBoxHSV.isChecked():
            self.progressBar.setValue(0)  # Réinitialiser pour chaque type de descripteur
            QtWidgets.QApplication.processEvents()
            features_hsv = self.loadFeatureType('HSV', 2)
            if features_hsv:  # Vérifier si des descripteurs ont été chargés
                self.features['HSV'] = features_hsv
                loaded_descriptors.append('HSV')
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger SIFT
        if self.checkBoxSIFT.isChecked():
            self.progressBar.setValue(0)  # Réinitialiser pour chaque type de descripteur
            QtWidgets.QApplication.processEvents()
            features_sift = self.loadFeatureType('SIFT', 3)
            if features_sift:  # Vérifier si des descripteurs ont été chargés
                self.features['SIFT'] = features_sift
                loaded_descriptors.append('SIFT')
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger ORB
        if self.checkBoxORB.isChecked():
            self.progressBar.setValue(0)  # Réinitialiser pour chaque type de descripteur
            QtWidgets.QApplication.processEvents()
            features_orb = self.loadFeatureType('ORB', 4)
            if features_orb:  # Vérifier si des descripteurs ont été chargés
                self.features['ORB'] = features_orb
                loaded_descriptors.append('ORB')
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger GLCM
        if self.checkBoxGLCM.isChecked():
            self.progressBar.setValue(0)  # Réinitialiser pour chaque type de descripteur
            QtWidgets.QApplication.processEvents()
            features_glcm = self.loadFeatureType('GLCM', 5)
            if features_glcm:  # Vérifier si des descripteurs ont été chargés
                self.features['GLCM'] = features_glcm
                loaded_descriptors.append('GLCM')
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger LBP
        if self.checkBoxLBP.isChecked():
            self.progressBar.setValue(0)  # Réinitialiser pour chaque type de descripteur
            QtWidgets.QApplication.processEvents()
            features_lbp = self.loadFeatureType('LBP', 6)
            if features_lbp:  # Vérifier si des descripteurs ont été chargés
                self.features['LBP'] = features_lbp
                loaded_descriptors.append('LBP')
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger HOG
        if self.checkBoxHOG.isChecked():
            self.progressBar.setValue(0)  # Réinitialiser pour chaque type de descripteur
            QtWidgets.QApplication.processEvents()
            features_hog = self.loadFeatureType('HOG', 7)
            if features_hog:  # Vérifier si des descripteurs ont été chargés
                self.features['HOG'] = features_hog
                loaded_descriptors.append('HOG')
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        self.progressBar.setValue(100)
        QtWidgets.QApplication.processEvents()
        
        # Compter le nombre d'images par classe
        self.countClassImages()
        
        # Afficher un message approprié en fonction des descripteurs chargés
        if loaded_descriptors:
            QtWidgets.QMessageBox.information(self, "Information", 
                                             f"Descripteurs chargés avec succès: {', '.join(loaded_descriptors)}")
        else:
            QtWidgets.QMessageBox.warning(self, "Attention", 
                                         "Aucun descripteur n'a pu être chargé. Veuillez indexer la base d'abord.")
    
    def countClassImages(self):
        """Compte le nombre d'images par classe dans la base"""
        self.class_counts = {}
        
        if os.path.exists(self.filenames):
            # Parcourir la structure de dossiers: animaux -> races -> images
            for animal_folder in os.listdir(self.filenames):
                animal_path = os.path.join(self.filenames, animal_folder)
                if os.path.isdir(animal_path):
                    for breed_folder in os.listdir(animal_path):
                        breed_path = os.path.join(animal_path, breed_folder)
                        if os.path.isdir(breed_path):
                            # Utiliser le chemin complet animal/race comme identifiant de classe
                            class_id = f"{animal_folder}/{breed_folder}"
                            
                            # Compter les images dans ce dossier
                            image_count = 0
                            for filename in os.listdir(breed_path):
                                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                    image_count += 1
                            
                            self.class_counts[class_id] = image_count
    
    def search(self):
        """Effectue la recherche d'images similaires"""
        # Vérifier qu'une image est chargée
        if not self.image_path:
            QtWidgets.QMessageBox.warning(self, "Attention", 
                                         "Veuillez d'abord charger une image requête.")
            return
        
        # Vérifier que des descripteurs sont chargés
        if not self.features:
            QtWidgets.QMessageBox.warning(self, "Attention", 
                                         "Veuillez d'abord charger les descripteurs.")
            return
        
        # Réinitialiser la barre de progression
        self.progressBar.setValue(0)
        
        # Afficher les descripteurs chargés pour le débogage
        print(f"Descripteurs chargés: {list(self.features.keys())}")
        for desc_type, features in self.features.items():
            print(f"Nombre de descripteurs {desc_type}: {len(features)}")
        
        # Déterminer le nombre d'images à afficher
        display_option = self.displayComboBox.currentText()
        if display_option == "Top 20":
            k = 20
        else:  # Top 50
            k = 50
        
        # Récupérer la distance sélectionnée
        distance_name = self.distanceComboBox.currentText()
        print(f"Distance sélectionnée: {distance_name}")
        
        # Nettoyer les résultats précédents
        self.clearResults()
        
        # Effectuer la recherche pour chaque type de descripteur
        combined_results = []
        total_descriptors = len(self.features)
        processed_descriptors = 0
        
        for desc_type, features in self.features.items():
            if not features:
                print(f"Aucun descripteur {desc_type} chargé")
                processed_descriptors += 1
                continue
            
            # Mettre à jour la barre de progression
            self.progressBar.setValue(int(100 * processed_descriptors / total_descriptors))
            QtWidgets.QApplication.processEvents()  # Forcer la mise à jour de l'interface
            
            # Déterminer l'algo_choice en fonction du type de descripteur
            if desc_type == 'BGR':
                algo_choice = 1
            elif desc_type == 'HSV':
                algo_choice = 2
            elif desc_type == 'SIFT':
                algo_choice = 3
            elif desc_type == 'ORB':
                algo_choice = 4
            elif desc_type == 'GLCM':
                algo_choice = 5
            elif desc_type == 'LBP':
                algo_choice = 6
            elif desc_type == 'HOG':
                algo_choice = 7
            else:
                processed_descriptors += 1
                continue
            
            print(f"Recherche avec {desc_type} (algo_choice={algo_choice})")
            
            # Extraire les caractéristiques de l'image requête
            try:
                req_features = extractReqFeatures(self.image_path, algo_choice)
                print(f"Caractéristiques extraites pour l'image requête: {req_features.shape if hasattr(req_features, 'shape') else 'None'}")
                
                # Adapter la distance pour ORB et SIFT
                current_distance = distance_name
                if desc_type == 'ORB' and distance_name != "Brute force":
                    current_distance = "Brute force"
                    print(f"Distance adaptée pour ORB: {current_distance}")
                elif desc_type == 'SIFT' and distance_name not in ["Euclidienne", "Flann"]:
                    current_distance = "Flann"
                    print(f"Distance adaptée pour SIFT: {current_distance}")
                
                # Rechercher les voisins
                neighbors = getkVoisins(features, req_features, k, current_distance)
                
                # Ajouter les résultats à la liste combinée
                for path, _, dist in neighbors:
                    combined_results.append((path, dist, desc_type))
                
                print(f"Recherche avec {desc_type} terminée: {len(neighbors)} résultats trouvés")
            except Exception as e:
                print(f"Erreur lors de la recherche avec {desc_type}: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Mettre à jour la barre de progression
            processed_descriptors += 1
            self.progressBar.setValue(int(100 * processed_descriptors / total_descriptors))
            QtWidgets.QApplication.processEvents()  # Forcer la mise à jour de l'interface
        
        # Trier les résultats combinés par distance
        if distance_name in ["Correlation", "Intersection"]:
            combined_results.sort(key=lambda x: -x[1])  # Tri décroissant pour les mesures de similarité
        else:
            combined_results.sort(key=lambda x: x[1])  # Tri croissant pour les mesures de distance
        
        # Prendre les k premiers résultats
        self.results = combined_results[:k]
        
        print(f"Nombre total de résultats combinés: {len(combined_results)}")
        print(f"Nombre de résultats après tri et limitation: {len(self.results)}")
        
        # Afficher les résultats
        self.displayResults()
        
        # Calculer et afficher les métriques
        if self.results:
            self.calculateMetrics()
        else:
            print("Aucun résultat à afficher, impossible de calculer les métriques")
            # Réinitialiser les métriques
            for i in range(5):
                self.metricsTable.setItem(i, 1, QtWidgets.QTableWidgetItem("N/A"))
        
        # Finaliser la barre de progression
        self.progressBar.setValue(100)
        
        print(f"Recherche terminée: {len(self.results)} résultats affichés")
    
    def clearResults(self):
        """Nettoie les résultats précédents"""
        # Supprimer tous les widgets du layout de résultats
        while self.resultsGrid.count():
            item = self.resultsGrid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
    
    def displayResults(self):
        """Affiche les résultats de la recherche"""
        # Déterminer le nombre de colonnes (5 par défaut)
        cols = 5
        
        # Afficher les images
        for i, (path, dist, desc_type) in enumerate(self.results):
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
                infoLabel = QtWidgets.QLabel(f"{os.path.basename(path)}\nDist: {dist:.4f}\nType: {desc_type}")
                infoLabel.setAlignment(QtCore.Qt.AlignCenter)
                layout.addWidget(infoLabel)
                
                # Ajouter le container au layout des résultats
                row = i // cols
                col = i % cols
                self.resultsGrid.addWidget(container, row, col)
                
            except Exception as e:
                print(f"Erreur lors de l'affichage de l'image {path}: {str(e)}")
    
    def calculateMetrics(self):
        """Calcule et affiche les métriques d'évaluation"""
        if not self.results:
            # Réinitialiser les métriques si aucun résultat
            for i in range(5):
                self.metricsTable.setItem(i, 1, QtWidgets.QTableWidgetItem("N/A"))
            return
        
        # Déterminer la classe de l'image requête
        req_path = self.image_path
        
        # Extraire l'animal et la race à partir du chemin
        path_parts = req_path.split(os.sep)
        
        # Trouver les indices de l'animal et de la race dans le chemin
        # Format attendu: .../MIR_DATASETS_B/animal/race/image.jpg
        animal_index = -1
        breed_index = -1
        
        for i, part in enumerate(path_parts):
            if part == "MIR_DATASETS_B" and i+2 < len(path_parts):
                animal_index = i+1
                breed_index = i+2
                break
        
        if animal_index >= 0 and breed_index >= 0:
            req_animal = path_parts[animal_index]
            req_breed = path_parts[breed_index]
            req_class = f"{req_animal}/{req_breed}"
            print(f"Classe de l'image requête identifiée: {req_class}")
        else:
            # Si l'image n'est pas dans la structure attendue, essayer d'extraire du nom de fichier
            file_name = os.path.basename(req_path)
            parts = os.path.splitext(file_name)[0].split('_')
            
            if len(parts) >= 2:
                req_animal = parts[0]
                req_breed = parts[1]
                req_class = f"{req_animal}/{req_breed}"
                print(f"Classe extraite du nom de fichier: {req_class}")
            else:
                QtWidgets.QMessageBox.warning(self, "Attention", 
                                             "Impossible de déterminer la classe de l'image requête.")
                return
        
        # Nombre total d'images pertinentes dans la base
        total_relevant = self.class_counts.get(req_class, 0)
        if total_relevant == 0:
            # Essayer de compter manuellement
            total_relevant = 0
            class_path = os.path.join(self.filenames, req_animal, req_breed)
            if os.path.exists(class_path):
                for file in os.listdir(class_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        total_relevant += 1
            
            if total_relevant == 0:
                QtWidgets.QMessageBox.warning(self, "Attention", 
                                             f"Aucune image de la classe {req_class} trouvée dans la base.")
                return
        
        print(f"Classe de l'image requête: {req_class}")
        print(f"Nombre total d'images pertinentes: {total_relevant}")
        
        # Calculer la pertinence de chaque résultat
        relevance = []
        for path, _, _ in self.results:
            try:
                # Extraire l'animal et la race à partir du chemin
                path_parts = path.split(os.sep)
                
                # Chercher les indices de l'animal et de la race
                animal_index = -1
                breed_index = -1
                
                for i, part in enumerate(path_parts):
                    if part == "MIR_DATASETS_B" and i+2 < len(path_parts):
                        animal_index = i+1
                        breed_index = i+2
                        break
                
                if animal_index >= 0 and breed_index >= 0:
                    img_animal = path_parts[animal_index]
                    img_breed = path_parts[breed_index]
                    img_class = f"{img_animal}/{img_breed}"
                    is_relevant = 1 if img_class == req_class else 0
                    relevance.append(is_relevant)
                    print(f"Image {path}: classe={img_class}, pertinence={is_relevant}")
                else:
                    relevance.append(0)
                    print(f"Image {path}: impossible de déterminer la classe")
            except Exception as e:
                relevance.append(0)
                print(f"Erreur lors de l'analyse du chemin {path}: {str(e)}")
        
        # Calculer le rappel et la précision à chaque position
        recalls = []
        precisions = []
        relevant_count = 0
        
        for i, rel in enumerate(relevance):
            if rel == 1:
                relevant_count += 1
            
            recall = relevant_count / total_relevant if total_relevant > 0 else 0
            precision = relevant_count / (i + 1) if i + 1 > 0 else 0
            
            recalls.append(recall)
            precisions.append(precision)
        
        # Calculer l'Average Precision (AP)
        ap = 0
        if sum(relevance) > 0:  # S'assurer qu'il y a au moins un document pertinent
            # Méthode 1: AP = somme(P(k) * rel(k)) / nombre de documents pertinents
            for i, rel in enumerate(relevance):
                if rel == 1:
                    ap += precisions[i]
            ap /= total_relevant
        
        # Calculer la R-Precision
        r_precision = 0
        if total_relevant <= len(relevance):
            r_precision = sum(relevance[:total_relevant]) / total_relevant if total_relevant > 0 else 0
        
        # Afficher les métriques dans le tableau
        self.metricsTable.setItem(0, 1, QtWidgets.QTableWidgetItem(f"{recalls[-1]:.4f}"))
        self.metricsTable.setItem(1, 1, QtWidgets.QTableWidgetItem(f"{precisions[-1]:.4f}"))
        self.metricsTable.setItem(2, 1, QtWidgets.QTableWidgetItem(f"{ap:.4f}"))
        self.metricsTable.setItem(3, 1, QtWidgets.QTableWidgetItem("N/A"))  # MAP nécessite plusieurs requêtes
        self.metricsTable.setItem(4, 1, QtWidgets.QTableWidgetItem(f"{r_precision:.4f}"))
        
        print(f"Rappel final: {recalls[-1]:.4f}")
        print(f"Précision finale: {precisions[-1]:.4f}")
        print(f"AP: {ap:.4f}")
        print(f"R-Precision: {r_precision:.4f}")
        
        # Tracer la courbe Rappel/Précision
        self.rpFigure.clear()
        ax = self.rpFigure.add_subplot(111)
        ax.plot(recalls, precisions, 'b-o')
        ax.set_xlabel('Rappel')
        ax.set_ylabel('Précision')
        ax.set_title('Courbe Rappel/Précision')
        ax.grid(True)
        self.rpCanvas.draw()
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
from descriptors import extractReqFeatures
from descriptors_page import showDialog
from metrics import MetricsWindow, calculate_metrics
from distances import getkVoisins
import statistics

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
        self.metrics_data = {}  # Pour stocker les métriques d'évaluation
        self.selected_descriptors = []  # Pour stocker les descripteurs sélectionnés
        
        # Variables pour les statistiques de temps
        self.search_times = []  # Liste des temps de recherche
        self.total_searches = 0  # Nombre total de recherches
        
        # Vérifier les descripteurs disponibles à l'initialisation
        self.checkAvailableDescriptors()
    
    def setupUi(self):
        self.setObjectName("SearchPage")
        self.resize(1200, 800)
        
        # Définir le titre de la fenêtre
        self.setWindowTitle("Moteur de Recherche d'Images")
        
        # Layout principal
        self.mainLayout = QtWidgets.QVBoxLayout(self)
        
        # Layout horizontal pour la partie supérieure (3 zones)
        self.topLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.addLayout(self.topLayout)
        
        # ZONE 1: Panneau de contrôle (descripteurs, distance, affichage)
        self.controlPanel = QtWidgets.QGroupBox("Contrôles")
        self.controlLayout = QtWidgets.QVBoxLayout(self.controlPanel)
        self.topLayout.addWidget(self.controlPanel)
        
        # Sélection des descripteurs
        self.descriptorsGroup = QtWidgets.QGroupBox("Descripteurs")
        self.descriptorsLayout = QtWidgets.QGridLayout(self.descriptorsGroup)  # Utiliser un GridLayout au lieu de QVBoxLayout
        self.controlLayout.addWidget(self.descriptorsGroup)
        
        # Checkboxes pour les descripteurs sur 2 colonnes
        self.checkBoxColor = QtWidgets.QCheckBox("BGR")
        self.descriptorsLayout.addWidget(self.checkBoxColor, 0, 0)
        
        self.checkBoxHSV = QtWidgets.QCheckBox("HSV")
        self.descriptorsLayout.addWidget(self.checkBoxHSV, 0, 1)
        
        self.checkBoxGLCM = QtWidgets.QCheckBox("GLCM")
        self.descriptorsLayout.addWidget(self.checkBoxGLCM, 1, 0)
        
        self.checkBoxHOG = QtWidgets.QCheckBox("HOG")
        self.descriptorsLayout.addWidget(self.checkBoxHOG, 1, 1)
        
        self.checkBoxLBP = QtWidgets.QCheckBox("LBP")
        self.descriptorsLayout.addWidget(self.checkBoxLBP, 2, 0)
        
        self.checkBoxORB = QtWidgets.QCheckBox("ORB")
        self.descriptorsLayout.addWidget(self.checkBoxORB, 2, 1)
        
        # Sélection de la distance
        self.distanceGroup = QtWidgets.QGroupBox("Distance")
        self.distanceLayout = QtWidgets.QVBoxLayout(self.distanceGroup)
        self.controlLayout.addWidget(self.distanceGroup)
        
        self.distanceComboBox = QtWidgets.QComboBox()
        self.distanceComboBox.addItems(["Cosinus", "Euclidienne", "Manhattan"])
        self.distanceLayout.addWidget(self.distanceComboBox)
        
        # ZONE 2: Boutons d'action
        self.buttonPanel = QtWidgets.QGroupBox("Actions")
        self.buttonLayout = QtWidgets.QVBoxLayout(self.buttonPanel)
        self.topLayout.addWidget(self.buttonPanel)
        
        self.loadFeaturesButton = QtWidgets.QPushButton("Charger les descripteurs")
        self.loadFeaturesButton.setMinimumHeight(15)
        self.buttonLayout.addWidget(self.loadFeaturesButton)
        
        self.loadImageButton = QtWidgets.QPushButton("Charger une image")
        self.loadImageButton.setMinimumHeight(15)
        self.buttonLayout.addWidget(self.loadImageButton)
        
        # Déplacer la section d'affichage ici
        self.displayGroup = QtWidgets.QGroupBox("Affichage")
        self.displayLayout = QtWidgets.QVBoxLayout(self.displayGroup)
        self.buttonLayout.addWidget(self.displayGroup)
        
        self.displayComboBox = QtWidgets.QComboBox()
        self.displayComboBox.addItems(["Top 20", "Top 50", "Top 100"])
        self.displayLayout.addWidget(self.displayComboBox)
        
        self.searchButton = QtWidgets.QPushButton("Rechercher")
        self.searchButton.setMinimumHeight(15)
        self.buttonLayout.addWidget(self.searchButton)
        
        # Ajouter le bouton des métriques ici
        self.metricsButton = QtWidgets.QPushButton("Voir les métriques")
        self.metricsButton.setMinimumHeight(15)
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
        
        # Bouton retour
        self.backButton = QtWidgets.QPushButton("Retour à l'accueil")
        self.backButton.setMinimumHeight(40)
        self.backButton.setProperty("class", "home-button")
        self.mainLayout.addWidget(self.backButton)
        
        # Connecter les signaux
        self.connectSignals()
    
    def connectSignals(self):
        """Connecte les signaux aux slots"""
        self.loadFeaturesButton.clicked.connect(self.loadDescriptors)
        self.loadImageButton.clicked.connect(self.loadImage)
        self.searchButton.clicked.connect(self.search)
        self.metricsButton.clicked.connect(self.showMetricsWindow)
        
        # Connecter les changements de descripteurs à la mise à jour des distances
        self.checkBoxColor.stateChanged.connect(self.updateDistanceOptions)
        self.checkBoxHSV.stateChanged.connect(self.updateDistanceOptions)
        self.checkBoxGLCM.stateChanged.connect(self.updateDistanceOptions)
        self.checkBoxHOG.stateChanged.connect(self.updateDistanceOptions)
        self.checkBoxLBP.stateChanged.connect(self.updateDistanceOptions)
        self.checkBoxORB.stateChanged.connect(self.updateDistanceOptions)
    
    def updateDistanceOptions(self):
        """Met à jour les options de distance en fonction des descripteurs sélectionnés"""
        self.distanceComboBox.clear()
        
        # Toujours ajouter la distance Euclidienne comme option par défaut
        self.distanceComboBox.addItem("Euclidienne")
        
        # Ajouter les distances communes pour la plupart des descripteurs
        self.distanceComboBox.addItems(["Manhattan", "Cosinus"])
        
        # Ajouter des distances spécifiques pour les descripteurs d'histogramme
        if any([self.checkBoxColor.isChecked(), self.checkBoxHSV.isChecked()]):
            self.distanceComboBox.addItems(["Chi carre", "Intersection", "Bhattacharyya", "Correlation"])
        
        # Ajouter des distances spécifiques pour ORB et SIFT
        if any([self.checkBoxORB.isChecked()]):
            self.distanceComboBox.addItem("Brute force")
            self.distanceComboBox.addItem("Flann")
        
        # Sélectionner Euclidienne par défaut
        index = self.distanceComboBox.findText("Euclidienne")
        if index >= 0:
            self.distanceComboBox.setCurrentIndex(index)
    
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
        
        # Construire le chemin complet vers le sous-dossier du descripteur
        folder_path = os.path.join('Descripteurs', folder_name)
        
        # Vérifier si le dossier existe
        if not os.path.exists(folder_path):
            print(f"Le dossier {folder_path} n'existe pas.")
            return []  # Retourner une liste vide sans afficher de message
        
        print(f"Chargement des descripteurs depuis {folder_path}")
        
        # Obtenir la liste de tous les fichiers .txt dans le dossier qui correspondent au type demandé
        all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt') and f.startswith(f"Methode_{algo_id}_")]
        
        # Si aucun fichier ne correspond au format spécifique, essayer de charger tous les fichiers .txt
        if not all_files:
            all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        
        total_files = len(all_files)
        
        if total_files == 0:
            print(f"Aucun fichier de descripteur trouvé dans {folder_path}")
            return []
        
        print(f"Nombre de fichiers trouvés: {total_files}")
        
        # Utiliser un dictionnaire pour accélérer la recherche d'images
        image_path_cache = {}
        
        # Traiter les fichiers par lots pour mettre à jour la barre de progression
        batch_size = max(1, total_files // 100)  # Diviser en environ 100 lots
        
        # Utiliser numpy pour charger les descripteurs plus rapidement
        for i in range(0, total_files, batch_size):
            # Traiter un lot de fichiers
            batch_files = all_files[i:i+batch_size]
            
            for file_name in batch_files:
                try:
                    # Charger le descripteur
                    data_path = os.path.join(folder_path, file_name)
                    feature = np.loadtxt(data_path)
                    
                    # Extraire les informations du nom de fichier
                    # Essayer différents formats possibles
                    parts = os.path.splitext(file_name)[0].split('_')
                    
                    # Format 1: Methode_1_animal_race_imagename.txt
                    if len(parts) >= 4 and parts[0] == "Methode":
                        animal = parts[2]
                        breed = parts[3]
                        image_name = '_'.join(parts[4:]) + '.jpg'
                    # Format 2: animal_race_imagename.txt
                    elif len(parts) >= 2:
                        animal = parts[0]
                        breed = parts[1]
                        image_name = '_'.join(parts[2:]) + '.jpg'
                    else:
                        print(f"Format de nom de fichier non reconnu: {file_name}")
                        continue
                    
                    # Construire le chemin de l'image
                    image_path = os.path.join(self.filenames, animal, breed, image_name)
                    
                    # Vérifier si le chemin existe déjà dans le cache
                    if image_name in image_path_cache:
                        image_path = image_path_cache[image_name]
                        features.append((image_path, feature))
                        continue
                    
                    if os.path.exists(image_path):
                        # Ajouter au cache
                        image_path_cache[image_name] = image_path
                        features.append((image_path, feature))
                    else:
                        # Essayer de trouver l'image avec la fonction find_image_in_directory
                        image_name_without_ext = os.path.splitext(image_name)[0]
                        found_path = self.find_image_in_directory(self.filenames, image_name_without_ext)
                        if found_path:
                            # Ajouter au cache
                            image_path_cache[image_name] = found_path
                            features.append((found_path, feature))
                        else:
                            print(f"Image introuvable même après recherche: {image_name}")
                
                except Exception as e:
                    print(f"Erreur lors du chargement de {file_name}: {str(e)}")
            
            # Mettre à jour la barre de progression
            progress_value = min(100, int(100 * (i + len(batch_files)) / total_files))
            self.progressBar.setValue(progress_value)
            QtWidgets.QApplication.processEvents()  # Forcer la mise à jour de l'interface
        
        print(f"Chargé {len(features)} descripteurs {folder_name}")
        return features

    def find_image_in_directory(self, base_dir, image_name):
        """
        Recherche efficacement une image dans la structure de dossiers.
        
        Args:
            base_dir: Dossier de base pour la recherche
            image_name: Nom de l'image à rechercher (sans extension)
            
        Returns:
            Chemin complet de l'image si trouvée, None sinon
        """
        # Liste des animaux et races pour la recherche rapide
        animaux = ["araignee", "chiens", "oiseaux", "poissons", "singes"]
        araignees = ["barn spider", "garden spider", "orb-weaving spider", "tarantula", "trap_door spider", "wolf spider"]
        chiens = ["boxer", "Chihuahua", "golden\x20retriever", "Labrador\x20retriever", "Rottweiler", "Siberian\x20husky"]
        oiseaux = ["blue jay", "bulbul", "great grey owl", "parrot", "robin", "vulture"]
        poissons = ["dogfish", "eagle ray", "guitarfish", "hammerhead", "ray", "tiger shark"]
        singes = ["baboon", "chimpanzee", "gorilla", "macaque", "orangutan", "squirrel monkey"]
        
        # Essayer différentes extensions
        for img_ext in ['.jpg', '.jpeg', '.png']:
            # Parcourir tous les animaux et races
            for animal in animaux:
                races_list = None
                if animal == "araignee":
                    races_list = araignees
                elif animal == "chiens":
                    races_list = chiens
                elif animal == "oiseaux":
                    races_list = oiseaux
                elif animal == "poissons":
                    races_list = poissons
                elif animal == "singes":
                    races_list = singes
                
                if races_list:
                    for race in races_list:
                        # Essayer de trouver l'image dans ce dossier
                        image_path = os.path.join(base_dir, animal, race, f"{image_name}{img_ext}")
                        if os.path.exists(image_path):
                            return image_path
        
        # Si on arrive ici, on n'a pas trouvé le fichier
        return None
    
    def loadDescriptors(self):
        """Charge les descripteurs sélectionnés"""
        # Vérifier qu'au moins un descripteur est sélectionné
        if not any([
            self.checkBoxColor.isChecked(),
            self.checkBoxHSV.isChecked(),
            self.checkBoxGLCM.isChecked(),
            self.checkBoxHOG.isChecked(),
            self.checkBoxLBP.isChecked(),
            self.checkBoxORB.isChecked()
        ]):
            showDialog()
            return
        
        # Commencer le chronométrage du chargement
        start_time = time.time()
        print("=" * 50)
        print("DÉBUT DU CHARGEMENT DES DESCRIPTEURS")
        print("=" * 50)
        
        self.features = {}  # Réinitialiser les descripteurs
        self.progressBar.setValue(0)
        QtWidgets.QApplication.processEvents()
        
        # Charger les descripteurs sélectionnés
        total_descriptors = sum([
            self.checkBoxColor.isChecked(),
            self.checkBoxHSV.isChecked(),
            self.checkBoxGLCM.isChecked(),
            self.checkBoxHOG.isChecked(),
            self.checkBoxLBP.isChecked(),
            self.checkBoxORB.isChecked()
        ])
        
        progress = 0
        loaded_descriptors = []
        
        # Charger histogramme de couleurs
        if self.checkBoxColor.isChecked():
            desc_start = time.time()
            self.progressBar.setValue(0)
            QtWidgets.QApplication.processEvents()
            features_color = self.loadFeatureType('BGR', 1)
            if features_color:
                self.features['BGR'] = features_color
                loaded_descriptors.append('BGR')
            desc_time = time.time() - desc_start
            print(f"Chargement BGR: {desc_time:.2f}s ({len(features_color) if features_color else 0} descripteurs)")
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger HOG
        if self.checkBoxHOG.isChecked():
            desc_start = time.time()
            self.progressBar.setValue(0)
            QtWidgets.QApplication.processEvents()
            features_hog = self.loadFeatureType('HOG', 2)
            if features_hog:
                self.features['HOG'] = features_hog
                loaded_descriptors.append('HOG')
            desc_time = time.time() - desc_start
            print(f"Chargement HOG: {desc_time:.2f}s ({len(features_hog) if features_hog else 0} descripteurs)")
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger LBP
        if self.checkBoxLBP.isChecked():
            desc_start = time.time()
            self.progressBar.setValue(0)
            QtWidgets.QApplication.processEvents()
            features_lbp = self.loadFeatureType('LBP', 3)
            if features_lbp:
                self.features['LBP'] = features_lbp
                loaded_descriptors.append('LBP')
            desc_time = time.time() - desc_start
            print(f"Chargement LBP: {desc_time:.2f}s ({len(features_lbp) if features_lbp else 0} descripteurs)")
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger ORB
        if self.checkBoxORB.isChecked():
            desc_start = time.time()
            self.progressBar.setValue(0)
            QtWidgets.QApplication.processEvents()
            features_orb = self.loadFeatureType('ORB', 4)
            if features_orb:
                self.features['ORB'] = features_orb
                loaded_descriptors.append('ORB')
            desc_time = time.time() - desc_start
            print(f"Chargement ORB: {desc_time:.2f}s ({len(features_orb) if features_orb else 0} descripteurs)")
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger Histogramme HSV
        if self.checkBoxHSV.isChecked():
            desc_start = time.time()
            self.progressBar.setValue(0)
            QtWidgets.QApplication.processEvents()
            features_hsv = self.loadFeatureType('HSV', 5)
            if features_hsv:
                self.features['HSV'] = features_hsv
                loaded_descriptors.append('HSV')
            desc_time = time.time() - desc_start
            print(f"Chargement HSV: {desc_time:.2f}s ({len(features_hsv) if features_hsv else 0} descripteurs)")
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger GLCM
        if self.checkBoxGLCM.isChecked():
            desc_start = time.time()
            self.progressBar.setValue(0)
            QtWidgets.QApplication.processEvents()
            features_glcm = self.loadFeatureType('GLCM', 6)
            if features_glcm:
                self.features['GLCM'] = features_glcm
                loaded_descriptors.append('GLCM')
            desc_time = time.time() - desc_start
            print(f"Chargement GLCM: {desc_time:.2f}s ({len(features_glcm) if features_glcm else 0} descripteurs)")
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Temps total de chargement
        total_loading_time = time.time() - start_time
        print("-" * 50)
        print(f"TEMPS TOTAL DE CHARGEMENT: {total_loading_time:.2f}s")
        print(f"Descripteurs chargés: {', '.join(loaded_descriptors)}")
        print("=" * 50)
        
        # Mettre à jour la liste des descripteurs sélectionnés
        self.selected_descriptors = loaded_descriptors
        
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
    
    def get_algo_choice(self, desc_type):
        """Retourne l'algo_choice correspondant au type de descripteur"""
        mapping = {
            'BGR': 1,
            'HOG': 2,
            'LBP': 3,
            'ORB': 4,
            'HSV': 5,
            'GLCM': 6
        }
        return mapping.get(desc_type, 0)

    def adapt_distance_for_descriptor(self, desc_type, distance_name):
        """Adapte la mesure de distance au type de descripteur"""
        if desc_type == 'ORB' and distance_name not in ["Brute force", "Flann"]:
            return "Brute force"
        elif desc_type in ['BGR', 'HSV'] and distance_name in ["Chi carre", "Intersection", "Bhattacharyya", "Correlation"]:
            return distance_name
        return distance_name
    
    def search(self):
        """Effectue la recherche d'images similaires avec combinaison de descripteurs"""
        # Commencer le chronométrage de la recherche
        search_start_time = time.time()
        self.total_searches += 1
        
        print("\n" + "=" * 50)
        print(f"DÉBUT DE LA RECHERCHE #{self.total_searches}")
        print("=" * 50)
        
        # Vérifier qu'une image est chargée
        if not self.image_path:
            QtWidgets.QMessageBox.warning(self, "Attention", 
                                         "Veuillez d'abord charger une image requête.")
            return
        
        # Vérifier que des descripteurs sont chargés et sélectionnés
        selected_descriptors = []
        if self.checkBoxColor.isChecked() and 'BGR' in self.features:
            selected_descriptors.append('BGR')
        if self.checkBoxHSV.isChecked() and 'HSV' in self.features:
            selected_descriptors.append('HSV')
        if self.checkBoxGLCM.isChecked() and 'GLCM' in self.features:
            selected_descriptors.append('GLCM')
        if self.checkBoxHOG.isChecked() and 'HOG' in self.features:
            selected_descriptors.append('HOG')
        if self.checkBoxLBP.isChecked() and 'LBP' in self.features:
            selected_descriptors.append('LBP')
        if self.checkBoxORB.isChecked() and 'ORB' in self.features:
            selected_descriptors.append('ORB')
        
        if not selected_descriptors:
            QtWidgets.QMessageBox.warning(self, "Attention", 
                                         "Veuillez sélectionner au moins un descripteur disponible.")
            return
        
        # Mémoriser les descripteurs sélectionnés
        self.selected_descriptors = selected_descriptors
        
        # Réinitialiser la barre de progression
        self.progressBar.setValue(0)
        
        # Réinitialiser les métriques
        self.metrics_data = {}
        self.metricsButton.setEnabled(False)
        
        # Déterminer le nombre d'images à afficher
        display_option = self.displayComboBox.currentText()
        k = 20 if display_option == "Top 20" else (50 if display_option == "Top 50" else 100)
        print(f"Nombre de résultats demandés: {k}")
        print(f"Descripteurs sélectionnés: {', '.join(selected_descriptors)}")
        
        # Récupérer la distance sélectionnée
        distance_name = self.distanceComboBox.currentText()
        print(f"Distance sélectionnée: {distance_name}")
        
        # Nettoyer les résultats précédents
        self.clearResults()
        
        # Dictionnaire pour stocker tous les résultats avec leurs scores
        all_results = {}  # {image_path: {desc_type: score, ...}, ...}
        
        # Pour normaliser les scores à la fin
        max_scores = {}  # {desc_type: max_score, ...}
        min_scores = {}  # {desc_type: min_score, ...}
        
        # Effectuer la recherche pour chaque descripteur sélectionné
        for desc_idx, desc_type in enumerate(selected_descriptors):
            desc_search_start = time.time()
            
            # Mettre à jour la barre de progression
            progress = int(100 * desc_idx / len(selected_descriptors))
            self.progressBar.setValue(progress)
            QtWidgets.QApplication.processEvents()
            # Extraire les caractéristiques de l'image requête
            try:
                # Déterminer l'algo_choice en fonction du type de descripteur
                algo_choice = self.get_algo_choice(desc_type)
                req_features = extractReqFeatures(self.image_path, algo_choice)
                
                # Pour tous les descripteurs, traitement normal
                if len(self.features[desc_type]) > 0:
                    sample_feature = self.features[desc_type][0][1]
                    if hasattr(req_features, 'shape') and hasattr(sample_feature, 'shape'):
                        if req_features.shape != sample_feature.shape:
                            print(f"Dimensions incompatibles: {req_features.shape} vs {sample_feature.shape}")
                            # Redimensionner le descripteur de la requête pour correspondre à la base
                            req_features = np.resize(req_features, sample_feature.shape)
                            print(f"Descripteur redimensionné à: {req_features.shape}")
                
                # Adapter la distance pour ce descripteur
                current_distance = self.adapt_distance_for_descriptor(desc_type, distance_name)
                
                # Rechercher les voisins
                neighbors = getkVoisins(self.features[desc_type], req_features, k, current_distance)
                
                # Initialiser les min/max pour ce descripteur
                if neighbors:
                    min_score = float('inf')
                    max_score = float('-inf')
                    
                    # Ajouter les résultats au dictionnaire global
                    for path, _, dist in neighbors:
                        if path not in all_results:
                            all_results[path] = {}
                        
                        # Stocker le score pour ce descripteur
                        score = dist
                        all_results[path][desc_type] = score
                        
                        # Mettre à jour min/max
                        min_score = min(min_score, score)
                        max_score = max(max_score, score)
                    
                    # Stocker min/max pour ce descripteur
                    min_scores[desc_type] = min_score
                    max_scores[desc_type] = max_score
                
                print(f"Recherche avec {desc_type}: {len(neighbors) if 'neighbors' in locals() else 0} résultats trouvés")
            except Exception as e:
                print(f"Erreur lors de la recherche avec {desc_type}: {str(e)}")
                import traceback
                traceback.print_exc()
            
            desc_search_time = time.time() - desc_search_start
            print(f"Recherche avec {desc_type}: {desc_search_time:.3f}s - {len(neighbors) if 'neighbors' in locals() else 0} résultats")
        
        # Combiner les scores pour tous les descripteurs
        combined_results = []
        
        for path, scores in all_results.items():
            # Calculer un score combiné normalisé
            combined_score = 0
            num_descriptors = 0
            contributed_descriptors = []
            
            for desc_type, score in scores.items():
                # Normaliser le score entre 0 et 1 (0 = meilleur, 1 = pire)
                min_score = min_scores.get(desc_type, 0)
                max_score = max_scores.get(desc_type, 1)
                score_range = max_score - min_score
                
                if score_range > 0:
                    # Pour les mesures de distance (plus petit = meilleur)
                    if distance_name not in ["Cosinus", "Correlation", "Intersection"]:
                        normalized_score = (score - min_score) / score_range
                    else:
                        # Pour les mesures de similarité (plus grand = meilleur)
                        normalized_score = 1 - (score - min_score) / score_range
                    
                    # Ajouter le score normalisé (sans pondération)
                    combined_score += normalized_score
                    num_descriptors += 1
                    contributed_descriptors.append(desc_type)
            
            # Calculer la moyenne des scores normalisés
            if num_descriptors > 0:
                avg_score = combined_score / num_descriptors
                combined_results.append((path, avg_score, "+".join(contributed_descriptors)))
        
        # Trier les résultats par score combiné
        combined_results.sort(key=lambda x: x[1])
        
        # Prendre les k premiers résultats
        self.results = combined_results[:k]
        
        # Afficher les résultats
        self.displayResults()
        
        # Finaliser la barre de progression
        self.progressBar.setValue(100)
        
        print(f"Recherche terminée: {len(self.results)} résultats affichés")
        
        # Temps total de recherche
        total_search_time = time.time() - search_start_time
        self.search_times.append(total_search_time)
        
        # Calculer le temps moyen
        average_search_time = statistics.mean(self.search_times)
        
        # Affichage des statistiques
        print("-" * 50)
        print(f"TEMPS DE RECHERCHE: {total_search_time:.3f}s")
        print(f"TEMPS MOYEN ({len(self.search_times)} recherches): {average_search_time:.3f}s")
        print(f"Résultats trouvés: {len(self.results)}")
        if len(self.search_times) > 1:
            min_time = min(self.search_times)
            max_time = max(self.search_times)
            print(f"Temps min/max: {min_time:.3f}s / {max_time:.3f}s")
        print("=" * 50)
    
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
                # Ajout d'une information sur le fait que c'est un score combiné
                dist_info = f"Score: {dist:.4f} (combiné)" if len(self.selected_descriptors) > 1 else f"Dist: {dist:.4f}"
                infoLabel = QtWidgets.QLabel(f"{os.path.basename(path)}\n{dist_info}\nType: {desc_type}")
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
        """Calcule les métriques d'évaluation pour la recherche actuelle"""
        if not self.results or not self.image_path:
            QtWidgets.QMessageBox.warning(
                self, 
                "Impossible de calculer les métriques", 
                "Veuillez d'abord effectuer une recherche."
            )
            return
        
        # Utiliser la fonction du module metrics
        self.metrics_data = calculate_metrics(self.results, self.image_path, self.class_counts)
    
    def showMetricsWindow(self):
        """Affiche la fenêtre des métriques d'évaluation"""
        self.calculateMetrics()
        metrics_window = MetricsWindow(self, self.metrics_data)
        metrics_window.exec_()

    def checkAvailableDescriptors(self):
        """Vérifie quels descripteurs sont disponibles et met à jour l'interface"""
        # Vérifier si le dossier Descripteurs existe
        if not os.path.exists('Descripteurs'):
            print("Le dossier Descripteurs n'existe pas")
            return
        
        # Liste des descripteurs et leurs checkboxes correspondantes (sans SIFT)
        descriptors_checkboxes = {
            'BGR': self.checkBoxColor,
            'HSV': self.checkBoxHSV,
            'GLCM': self.checkBoxGLCM,
            'HOG': self.checkBoxHOG,
            'LBP': self.checkBoxLBP,
            'ORB': self.checkBoxORB
        }
        
        # Parcourir les sous-dossiers de Descripteurs
        available_descriptors = os.listdir('Descripteurs')
        print(f"Descripteurs disponibles: {available_descriptors}")
        
        # Mettre à jour l'apparence des checkboxes
        for desc_name, checkbox in descriptors_checkboxes.items():
            if desc_name in available_descriptors:
                # Vérifier si le dossier contient des fichiers .txt
                desc_folder = os.path.join('Descripteurs', desc_name)
                if os.path.isdir(desc_folder) and any(f.endswith('.txt') for f in os.listdir(desc_folder)):
                    # Descripteur disponible et indexé
                    checkbox.setEnabled(True)
                    checkbox.setStyleSheet("")  # Style normal
                    checkbox.setToolTip(f"Descripteur {desc_name} disponible")
                else:
                    # Dossier existe mais pas de fichiers .txt
                    checkbox.setEnabled(False)
                    checkbox.setStyleSheet("color: rgba(0, 0, 0, 128);")  # Semi-transparent
                    checkbox.setToolTip(f"Descripteur {desc_name} non indexé")
            else:
                # Descripteur non disponible
                checkbox.setEnabled(False)
                checkbox.setStyleSheet("color: rgba(0, 0, 0, 128);")  # Semi-transparent
                checkbox.setToolTip(f"Descripteur {desc_name} non disponible")
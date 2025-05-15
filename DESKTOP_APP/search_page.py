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
        self.metrics_data = {}  # Pour stocker les métriques d'évaluation
        self.selected_descriptors = []  # Pour stocker les descripteurs sélectionnés
        
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
        
        self.checkBoxSIFT = QtWidgets.QCheckBox("SIFT")
        self.descriptorsLayout.addWidget(self.checkBoxSIFT, 3, 0)
        
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
        self.displayComboBox.addItems(["Top 20", "Top 50"])
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
        
        # Bouton Retour en bas de la page
        self.backButton = QtWidgets.QPushButton("Retour à l'accueil")
        self.backButton.setMinimumHeight(30)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.backButton.setFont(font)
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
        self.checkBoxSIFT.stateChanged.connect(self.updateDistanceOptions)
    
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
        if any([self.checkBoxORB.isChecked(), self.checkBoxSIFT.isChecked()]):
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
            print(f"Aucun fichier au format Methode_{algo_id}_*.txt trouvé, chargement de tous les fichiers .txt")
        
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
            self.checkBoxORB.isChecked(),
            self.checkBoxSIFT.isChecked()
        ]):
            showDialog()
            return
        
        self.features = {}  # Réinitialiser les descripteurs
        self.progressBar.setValue(0)
        QtWidgets.QApplication.processEvents()  # Forcer la mise à jour de l'interface
        
        # Charger les descripteurs sélectionnés
        total_descriptors = sum([
            self.checkBoxColor.isChecked(),
            self.checkBoxHSV.isChecked(),
            self.checkBoxGLCM.isChecked(),
            self.checkBoxHOG.isChecked(),
            self.checkBoxLBP.isChecked(),
            self.checkBoxORB.isChecked(),
            self.checkBoxSIFT.isChecked()
        ])
        
        progress = 0
        loaded_descriptors = []
        
        # Charger histogramme de couleurs
        if self.checkBoxColor.isChecked():
            self.progressBar.setValue(0)  # Réinitialiser pour chaque type de descripteur
            QtWidgets.QApplication.processEvents()
            features_color = self.loadFeatureType('BGR', 1)
            if features_color:  # Vérifier si des descripteurs ont été chargés
                self.features['BGR'] = features_color
                loaded_descriptors.append('BGR')
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger HOG
        if self.checkBoxHOG.isChecked():
            self.progressBar.setValue(0)  # Réinitialiser pour chaque type de descripteur
            QtWidgets.QApplication.processEvents()
            features_hog = self.loadFeatureType('HOG', 2)
            if features_hog:  # Vérifier si des descripteurs ont été chargés
                self.features['HOG'] = features_hog
                loaded_descriptors.append('HOG')
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger LBP
        if self.checkBoxLBP.isChecked():
            self.progressBar.setValue(0)  # Réinitialiser pour chaque type de descripteur
            QtWidgets.QApplication.processEvents()
            features_lbp = self.loadFeatureType('LBP', 3)
            if features_lbp:  # Vérifier si des descripteurs ont été chargés
                self.features['LBP'] = features_lbp
                loaded_descriptors.append('LBP')
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
        
        # Charger Histogramme HSV
        if self.checkBoxHSV.isChecked():
            self.progressBar.setValue(0)  # Réinitialiser pour chaque type de descripteur
            QtWidgets.QApplication.processEvents()
            features_hsv = self.loadFeatureType('HSV', 5)
            if features_hsv:  # Vérifier si des descripteurs ont été chargés
                self.features['HSV'] = features_hsv
                loaded_descriptors.append('HSV')
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger GLCM
        if self.checkBoxGLCM.isChecked():
            self.progressBar.setValue(0)  # Réinitialiser pour chaque type de descripteur
            QtWidgets.QApplication.processEvents()
            features_glcm = self.loadFeatureType('GLCM', 6)
            if features_glcm:  # Vérifier si des descripteurs ont été chargés
                self.features['GLCM'] = features_glcm
                loaded_descriptors.append('GLCM')
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        # Charger SIFT
        if self.checkBoxSIFT.isChecked():
            self.progressBar.setValue(0)  # Réinitialiser pour chaque type de descripteur
            QtWidgets.QApplication.processEvents()
            features_sift = self.loadFeatureType('SIFT', 7)
            if features_sift:  # Vérifier si des descripteurs ont été chargés
                self.features['SIFT'] = features_sift
                loaded_descriptors.append('SIFT')
            progress += 1
            self.progressBar.setValue(int(100 * progress / total_descriptors))
            QtWidgets.QApplication.processEvents()
        
        self.progressBar.setValue(100)
        QtWidgets.QApplication.processEvents()
        
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
            'GLCM': 6,
            'SIFT': 7
        }
        return mapping.get(desc_type, 0)

    def adapt_distance_for_descriptor(self, desc_type, distance_name):
        """Adapte la mesure de distance au type de descripteur"""
        if desc_type in ['ORB', 'SIFT'] and distance_name not in ["Brute force", "Flann"]:
            return "Brute force"
        elif desc_type in ['BGR', 'HSV'] and distance_name in ["Chi carre", "Intersection", "Bhattacharyya", "Correlation"]:
            return distance_name
        return distance_name
    
    def search(self):
        """Effectue la recherche d'images similaires avec combinaison de descripteurs"""
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
        if self.checkBoxSIFT.isChecked() and 'SIFT' in self.features:
            selected_descriptors.append('SIFT')
        
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
        k = 20 if display_option == "Top 20" else 50
        
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
            # Mettre à jour la barre de progression
            progress = int(100 * desc_idx / len(selected_descriptors))
            self.progressBar.setValue(progress)
            QtWidgets.QApplication.processEvents()
            # Extraire les caractéristiques de l'image requête
            try:
                # Déterminer l'algo_choice en fonction du type de descripteur
                algo_choice = self.get_algo_choice(desc_type)
                req_features = extractReqFeatures(self.image_path, algo_choice)
                
                # Vérifier si les dimensions sont compatibles avec les descripteurs de la base
                if len(self.features[desc_type]) > 0:
                    sample_feature = self.features[desc_type][0][1]  # Prendre le premier descripteur comme exemple
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
                
                print(f"Recherche avec {desc_type} terminée: {len(neighbors)} résultats trouvés")
            except Exception as e:
                print(f"Erreur lors de la recherche avec {desc_type}: {str(e)}")
                import traceback
                traceback.print_exc()
        
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
        metrics_window = MetricsWindow(self, self.metrics_data)
        metrics_window.exec_()

    def checkAvailableDescriptors(self):
        """Vérifie quels descripteurs sont disponibles et met à jour l'interface"""
        # Vérifier si le dossier Descripteurs existe
        if not os.path.exists('Descripteurs'):
            print("Le dossier Descripteurs n'existe pas")
            return
        
        # Liste des descripteurs et leurs checkboxes correspondantes
        descriptors_checkboxes = {
            'BGR': self.checkBoxColor,
            'HSV': self.checkBoxHSV,
            'GLCM': self.checkBoxGLCM,
            'HOG': self.checkBoxHOG,
            'LBP': self.checkBoxLBP,
            'ORB': self.checkBoxORB,
            'SIFT': self.checkBoxSIFT
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

class MetricsWindow(QtWidgets.QDialog):
    def __init__(self, parent=None, metrics_data=None):
        super(MetricsWindow, self).__init__(parent)
        self.setWindowTitle("Métriques d'Évaluation")
        self.resize(600, 500)
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
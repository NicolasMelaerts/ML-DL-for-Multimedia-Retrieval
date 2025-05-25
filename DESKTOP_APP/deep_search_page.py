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
from search_page import MetricsWindow
from metrics import calculate_metrics

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
        
        # Définir le titre de la fenêtre
        self.setWindowTitle("Moteur de Recherche d'Images par Deep Learning")
        
        # Layout principal
        self.mainLayout = QtWidgets.QVBoxLayout(self)
        
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
        self.metricsButton.setEnabled(False)  # Désactivé jusqu'à ce qu'une recherche soit effectuée
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
        import time
        total_start_time = time.time()
        
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
        
        print("\n--- MESURE DES PERFORMANCES DE CHARGEMENT ---")
        
        try:
            for i, model_name in enumerate(selected_models):
                # Afficher le modèle en cours de chargement
                self.progressBar.setFormat(f"Préparation du modèle {model_name}: %p%")
                self.progressBar.setValue(0)
                QtWidgets.QApplication.processEvents()
                
                # Mesurer le temps de chargement pour ce modèle
                model_start_time = time.time()
                
                # Charger les features du modèle
                features_dict, image_dict = self.load_features_with_images(
                    f"./Features/{model_name}", self.filenames)
                
                model_time = time.time() - model_start_time
                print(f"Modèle {model_name}: {model_time:.4f} secondes")
                
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
            
            total_time = time.time() - total_start_time
            print(f"Temps total de chargement: {total_time:.4f} secondes")
            print("--------------------------------\n")
    
    def load_features_with_images(self, feature_folder, image_folder):
        """
        Charge les features et trouve les images correspondantes de manière optimisée.
        
        Args:
            feature_folder: Dossier contenant les fichiers de features
            image_folder: Dossier contenant les images
            
        Returns:
            Tuple (features_dict, image_dict) contenant les features et les chemins des images
        """
        import time
        start_time = time.time()
        
        feature_files = sorted(glob.glob(os.path.join(feature_folder, "*.txt")))
        features_dict = {}
        image_dict = {}
        
        total_files = len(feature_files)
        print(f"Chargement de {total_files} fichiers depuis {feature_folder}")
        
        # Initialiser la barre de progression pour ce modèle
        self.progressBar.setFormat(f"Chargement de {os.path.basename(feature_folder)}: %p%")
        
        # Créer un cache de correspondance entre les noms de fichiers et les chemins d'images
        # Cette étape préliminaire accélère considérablement la recherche d'images
        print("Création du cache de correspondance...")
        cache_start_time = time.time()
        image_cache = {}
        
        # Définir les listes d'animaux et de races exactement comme dans text_search.py
        animaux = ["araignee", "chiens", "oiseaux", "poissons", "singes"]
        araignees = ["barn spider", "garden spider", "orb-weaving spider", "tarantula", "trap_door spider", "wolf spider"]
        chiens = ["boxer", "Chihuahua", "golden\x20retriever", "Labrador\x20retriever", "Rottweiler", "Siberian\x20husky"]
        oiseaux = ["blue jay", "bulbul", "great grey owl", "parrot", "robin", "vulture"]
        poissons = ["dogfish", "eagle ray", "guitarfish", "hammerhead", "ray", "tiger shark"]
        singes = ["baboon", "chimpanzee", "gorilla", "macaque", "orangutan", "squirrel monkey"]
        
        races_dict = {
            "araignee": araignees,
            "chiens": chiens,
            "oiseaux": oiseaux,
            "poissons": poissons,
            "singes": singes
        }
        
        # Pré-construire un cache des chemins d'images
        for animal in animaux:
            animal_dir = os.path.join(image_folder, animal)
            if not os.path.isdir(animal_dir):
                print(f"Dossier animal non trouvé: {animal_dir}")
                continue
            
            for race in races_dict[animal]:
                race_dir = os.path.join(animal_dir, race)
                if not os.path.isdir(race_dir):
                    # Essayer avec des underscores à la place des espaces
                    race_dir = os.path.join(animal_dir, race.replace(' ', '_'))
                    if not os.path.isdir(race_dir):
                        print(f"Dossier race non trouvé: {race_dir}")
                        continue
            
                print(f"Indexation des images dans {race_dir}")
                try:
                    for file_name in os.listdir(race_dir):
                        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            base_name = os.path.splitext(file_name)[0]
                            image_cache[base_name] = os.path.join(race_dir, file_name)
                except Exception as e:
                    print(f"Erreur lors de l'indexation de {race_dir}: {str(e)}")
        
        cache_time = time.time() - cache_start_time
        print(f"Cache créé en {cache_time:.4f} secondes avec {len(image_cache)} entrées")
        
        # Mesurer les différentes étapes
        feature_loading_time = 0
        image_lookup_time = 0
        
        for i, file in enumerate(feature_files):
            try:
                # Mettre à jour la barre de progression
                progress = int(100 * i / total_files)
                self.progressBar.setValue(progress)
                if i % 100 == 0:  # Mettre à jour l'interface moins fréquemment pour accélérer
                    QtWidgets.QApplication.processEvents()
                
                # Charger le vecteur de caractéristiques
                feature_start = time.time()
                feature_vector = np.loadtxt(file, ndmin=1)
                base_name = os.path.splitext(os.path.basename(file))[0]
                feature_loading_time += time.time() - feature_start
                
                # Stocker les features
                features_dict[base_name] = feature_vector
                
                # Rechercher l'image dans le cache
                lookup_start = time.time()
                if base_name in image_cache:
                    image_dict[base_name] = image_cache[base_name]
                else:
                    # Si pas dans le cache, essayer de trouver directement
                    for ext in ['.jpg', '.jpeg', '.png']:
                        direct_path = os.path.join(image_folder, base_name + ext)
                        if os.path.exists(direct_path):
                            image_dict[base_name] = direct_path
                            break
                
                    # Si toujours pas trouvé, essayer de trouver l'image en parcourant tous les dossiers
                    if base_name not in image_dict:
                        # Extraire les parties du nom de fichier (format attendu: X_Y_animal_race_ZZZZ)
                        parts = base_name.split('_')
                        if len(parts) >= 5:
                            animal_name = parts[2]
                            race_name = parts[3]
                            
                            # Vérifier si l'animal est dans notre liste
                            if animal_name in animaux:
                                # Déterminer la liste de races correspondante
                                races_list = races_dict.get(animal_name, [])
                                
                                # Trouver la race correspondante
                                matching_race = None
                                for race in races_list:
                                    race_lower = race.lower()
                                    race_name_lower = race_name.lower()
                                    
                                    if (race_lower == race_name_lower or 
                                        race_lower.replace(' ', '_') == race_name_lower or
                                        race_lower.replace(' ', '') == race_name_lower):
                                        matching_race = race
                                        break
                                
                                if matching_race:
                                    # Essayer avec le nom exact de la race
                                    for ext in ['.jpg', '.jpeg', '.png']:
                                        path = os.path.join(image_folder, animal_name, matching_race, base_name + ext)
                                        if os.path.exists(path):
                                            image_dict[base_name] = path
                                            break
                                    
                                    # Essayer aussi avec des underscores
                                    path = os.path.join(image_folder, animal_name, matching_race.replace(' ', '_'), base_name + ext)
                                    if os.path.exists(path):
                                        image_dict[base_name] = path
                                        break
            
                image_lookup_time += time.time() - lookup_start
                
            except Exception as e:
                print(f"Erreur lors du chargement de {file}: {str(e)}")
        
        # Finaliser la barre de progression pour ce modèle
        self.progressBar.setValue(100)
        self.progressBar.setFormat("%p%")  # Remettre le format par défaut
        
        # Compter les images par classe pour les métriques
        self.class_counts = self.count_images_by_class(image_dict)
        
        total_time = time.time() - start_time
        print(f"Chargement des features: {feature_loading_time:.4f} secondes")
        print(f"Recherche des images: {image_lookup_time:.4f} secondes")
        print(f"Temps total pour {feature_folder}: {total_time:.4f} secondes")
        print(f"{len(features_dict)} caractéristiques chargées avec {len(image_dict)} images")
        
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
        import time
        total_start_time = time.time()
        
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
        
        print("\n--- MESURE DES PERFORMANCES DE RECHERCHE PROFONDE ---")
        print(f"Image requête: {self.image_path}")
        print(f"Nom de l'image requête: {self.query_name}")
        
        # Effectuer la recherche pour chaque modèle sélectionné
        search_start_time = time.time()
        total_models = len(self.features_dict)
        model_times = {}
        
        for i, (model_name, features_dict) in enumerate(self.features_dict.items()):
            model_start_time = time.time()
            
            # Mettre à jour la barre de progression
            progress = int((i / total_models) * 50)  # 50% pour la recherche
            self.progressBar.setValue(progress)
            QtWidgets.QApplication.processEvents()
            
            # Vérifier si l'image requête a des features pour ce modèle
            if self.query_name in features_dict:
                print(f"Image requête trouvée dans les features de {model_name}")
                # Rechercher les k plus proches voisins
                knn_start_time = time.time()
                neighbors = getkVoisins_deep(features_dict, self.query_name, k)
                knn_time = time.time() - knn_start_time
                
                print(f"Nombre de voisins trouvés pour {model_name}: {len(neighbors)}")
                
                # Ajouter les résultats à la liste globale
                image_lookup_time = 0
                found_count = 0
                for neighbor_name, dist in neighbors:
                    img_start_time = time.time()
                    if neighbor_name in self.image_dict:
                        self.results.append((self.image_dict[neighbor_name], dist, model_name))
                        found_count += 1
                    else:
                        print(f"Image non trouvée dans le dictionnaire: {neighbor_name}")
                    image_lookup_time += time.time() - img_start_time
                
                print(f"Images trouvées dans le dictionnaire pour {model_name}: {found_count}/{len(neighbors)}")
            else:
                print(f"L'image requête n'a pas de features pour le modèle {model_name}")
                knn_time = 0
                image_lookup_time = 0
            
            model_time = time.time() - model_start_time
            model_times[model_name] = {
                "total": model_time,
                "knn": knn_time,
                "image_lookup": image_lookup_time
            }
        
        search_time = time.time() - search_start_time
        print(f"1. Recherche dans les modèles: {search_time:.4f} secondes")
        for model_name, times in model_times.items():
            print(f"   - {model_name}: {times['total']:.4f} secondes (KNN: {times['knn']:.4f}s, Image lookup: {times['image_lookup']:.4f}s)")
        
        # Trier les résultats par distance (croissante)
        sort_start_time = time.time()
        self.results.sort(key=lambda x: x[1])
        sort_time = time.time() - sort_start_time
        print(f"2. Tri des résultats: {sort_time:.4f} secondes")
        print(f"Nombre total de résultats avant limitation: {len(self.results)}")
        
        # Limiter aux k premiers résultats
        self.results = self.results[:k]
        print(f"Nombre de résultats après limitation à {k}: {len(self.results)}")
        
        # Effacer les résultats précédents
        clear_start_time = time.time()
        self.clearResults()
        clear_time = time.time() - clear_start_time
        print(f"3. Effacement des résultats précédents: {clear_time:.4f} secondes")
        
        # Afficher les nouveaux résultats
        display_start_time = time.time()
        self.displayResults()
        display_time = time.time() - display_start_time
        print(f"4. Affichage des résultats: {display_time:.4f} secondes")
        
        # Calculer les métriques
        metrics_start_time = time.time()
        self.calculateMetrics()
        metrics_time = time.time() - metrics_start_time
        print(f"5. Calcul des métriques: {metrics_time:.4f} secondes")
        
        # Finaliser la barre de progression
        self.progressBar.setValue(100)
        
        total_time = time.time() - total_start_time
        print(f"Temps total de recherche: {total_time:.4f} secondes")
        print("--------------------------------\n")
        
        print(f"Recherche terminée: {len(self.results)} résultats affichés")
        
        # Activer le bouton des métriques après la recherche
        self.metricsButton.setEnabled(True)
    
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
                    print(f"Impossible de charger l'image: {path}")
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

    def count_classes(self):
        """Compte le nombre d'images par classe dans la base de données"""
        self.class_counts = {}
        base_dir = "MIR_DATASETS_B"  # Ajuster le chemin si nécessaire
        
        if os.path.exists(base_dir):
            for animal_dir in os.listdir(base_dir):
                animal_path = os.path.join(base_dir, animal_dir)
                if os.path.isdir(animal_path):
                    for breed_dir in os.listdir(animal_path):
                        breed_path = os.path.join(animal_path, breed_dir)
                        if os.path.isdir(breed_path):
                            class_key = f"{animal_dir}/{breed_dir}"
                            image_count = len([f for f in os.listdir(breed_path) 
                                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                            self.class_counts[class_key] = image_count 
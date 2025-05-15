# -*- coding: utf-8 -*-
"""
Page de recherche d'images par texte
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import time

class TextSearchPage(QtWidgets.QWidget):
    def __init__(self, parent=None, model_path="Transformer/sentence_transformer_model"):
        super(TextSearchPage, self).__init__(parent)
        self.model_path = model_path
        self.model = None
        self.captions = {}
        
        # Configuration des animaux et races pour la recherche rapide d'images
        self.animaux = ["araignee", "chiens", "oiseaux", "poissons", "singes"]
        self.araignees = ["barn spider", "garden spider", "orb-weaving spider", "tarantula", "trap_door spider", "wolf spider"]
        self.chiens = ["boxer", "Chihuahua", "golden\x20retriever", "Labrador\x20retriever", "Rottweiler", "Siberian\x20husky"]
        self.oiseaux = ["blue jay", "bulbul", "great grey owl", "parrot", "robin", "vulture"]
        self.poissons = ["dogfish", "eagle ray", "guitarfish", "hammerhead", "ray", "tiger shark"]
        self.singes = ["baboon", "chimpanzee", "gorilla", "macaque", "orangutan", "squirrel monkey"]
        
        self.setupUi()
        
    def setupUi(self):
        self.setObjectName("TextSearchPage")
        self.resize(1000, 800)
        
        # Définir le titre de la fenêtre
        self.setWindowTitle("Recherche d'Images par Texte")
        
        # Layout principal
        self.mainLayout = QtWidgets.QVBoxLayout(self)
        
        # Espace
        self.mainLayout.addSpacing(5)
        
        # Section de configuration (simple, non collapsible)
        self.configGroupBox = QtWidgets.QGroupBox("Configuration")
        self.configLayout = QtWidgets.QVBoxLayout(self.configGroupBox)
        self.configLayout.setContentsMargins(10, 20, 10, 10)  # Marges réduites
        self.configLayout.setSpacing(5)  # Espacement réduit
        
        # Layout horizontal pour tous les champs de configuration
        self.configFieldsLayout = QtWidgets.QHBoxLayout()
        
        # Première colonne: JSON et modèle
        self.configCol1Layout = QtWidgets.QVBoxLayout()
        
        # Sélection du fichier JSON (simplifié)
        self.jsonLayout = QtWidgets.QHBoxLayout()
        self.jsonLabel = QtWidgets.QLabel("JSON:")
        self.jsonEdit = QtWidgets.QLineEdit("Transformer/captions.json")
        self.jsonButton = QtWidgets.QPushButton("...")
        self.jsonButton.setMaximumWidth(30)
        self.jsonLayout.addWidget(self.jsonLabel)
        self.jsonLayout.addWidget(self.jsonEdit)
        self.jsonLayout.addWidget(self.jsonButton)
        self.configCol1Layout.addLayout(self.jsonLayout)
        
        # Sélection du modèle (simplifié)
        self.modelLayout = QtWidgets.QHBoxLayout()
        self.modelLabel = QtWidgets.QLabel("Modèle:")
        self.modelEdit = QtWidgets.QLineEdit(self.model_path)
        self.modelButton = QtWidgets.QPushButton("...")
        self.modelButton.setMaximumWidth(30)
        self.modelLayout.addWidget(self.modelLabel)
        self.modelLayout.addWidget(self.modelEdit)
        self.modelLayout.addWidget(self.modelButton)
        self.configCol1Layout.addLayout(self.modelLayout)
        
        # Ajouter la première colonne au layout des champs
        self.configFieldsLayout.addLayout(self.configCol1Layout)
        
        # Deuxième colonne: Embeddings et bouton de chargement
        self.configCol2Layout = QtWidgets.QVBoxLayout()
        
        # Sélection du dossier d'embeddings (simplifié)
        self.embeddingsLayout = QtWidgets.QHBoxLayout()
        self.embeddingsLabel = QtWidgets.QLabel("Embeddings:")
        self.embeddingsEdit = QtWidgets.QLineEdit("Transformer/embeddings_output")
        self.embeddingsButton = QtWidgets.QPushButton("...")
        self.embeddingsButton.setMaximumWidth(30)
        self.embeddingsLayout.addWidget(self.embeddingsLabel)
        self.embeddingsLayout.addWidget(self.embeddingsEdit)
        self.embeddingsLayout.addWidget(self.embeddingsButton)
        self.configCol2Layout.addLayout(self.embeddingsLayout)
        
        # Bouton de chargement
        self.loadButton = QtWidgets.QPushButton("Charger le modèle")
        self.configCol2Layout.addWidget(self.loadButton)
        
        # Ajouter la deuxième colonne au layout des champs
        self.configFieldsLayout.addLayout(self.configCol2Layout)
        
        # Ajouter le layout des champs au layout de configuration
        self.configLayout.addLayout(self.configFieldsLayout)
        
        # Barre de progression
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setValue(0)
        self.configLayout.addWidget(self.progressBar)
        
        # Ajouter la section de configuration au layout principal avec une taille fixe
        self.mainLayout.addWidget(self.configGroupBox)
        
        # Section de recherche (plus compacte)
        self.searchGroupBox = QtWidgets.QGroupBox("Recherche")
        self.searchGroupBox.setEnabled(False)  # Désactivé jusqu'au chargement du modèle
        self.searchLayout = QtWidgets.QHBoxLayout(self.searchGroupBox)
        self.searchLayout.setContentsMargins(10, 20, 10, 10)  # Marges réduites
        
        # Layout pour le champ de recherche
        self.searchInputLayout = QtWidgets.QVBoxLayout()
        
        # Champ de recherche
        self.searchInputLayout.addWidget(QtWidgets.QLabel("Description textuelle:"))
        self.searchEdit = QtWidgets.QLineEdit()
        self.searchEdit.setPlaceholderText("Entrez une description d'image (ex: 'a bird standing on the ground')")
        self.searchInputLayout.addWidget(self.searchEdit)
        
        # Ajouter le layout du champ de recherche au layout de recherche
        self.searchLayout.addLayout(self.searchInputLayout, 3)  # Proportion 3
        
        # Layout pour les options et le bouton
        self.searchOptionsLayout = QtWidgets.QVBoxLayout()
        
        # Nombre de résultats
        self.resultsLayout = QtWidgets.QHBoxLayout()
        self.resultsLabel = QtWidgets.QLabel("Nombre de résultats:")
        self.resultsSpinBox = QtWidgets.QSpinBox()
        self.resultsSpinBox.setMinimum(1)
        self.resultsSpinBox.setMaximum(20)
        self.resultsSpinBox.setValue(5)
        self.resultsLayout.addWidget(self.resultsLabel)
        self.resultsLayout.addWidget(self.resultsSpinBox)
        self.searchOptionsLayout.addLayout(self.resultsLayout)
        
        # Bouton de recherche
        self.searchButton = QtWidgets.QPushButton("Rechercher")
        self.searchButton.setMinimumHeight(40)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.searchButton.setFont(font)
        self.searchOptionsLayout.addWidget(self.searchButton)
        
        # Ajouter le layout des options au layout de recherche
        self.searchLayout.addLayout(self.searchOptionsLayout, 1)  # Proportion 1
        
        # Ajouter la section de recherche au layout principal
        self.mainLayout.addWidget(self.searchGroupBox)
        
        # Section des résultats (agrandie)
        self.resultsGroupBox = QtWidgets.QGroupBox("Résultats")
        self.resultsMainLayout = QtWidgets.QVBoxLayout(self.resultsGroupBox)
        
        # Zone de défilement pour les résultats
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.resultsMainLayout.addWidget(self.scrollArea)
        
        # Ajouter la section des résultats au layout principal avec une proportion plus grande
        self.mainLayout.addWidget(self.resultsGroupBox, 1)  # Stretch factor de 1
        
        # Bouton de retour
        self.backButton = QtWidgets.QPushButton("Retour à l'accueil")
        self.mainLayout.addWidget(self.backButton)
        
        # Connecter les signaux
        self.jsonButton.clicked.connect(self.browse_json)
        self.modelButton.clicked.connect(self.browse_model)
        self.embeddingsButton.clicked.connect(self.browse_embeddings)
        self.loadButton.clicked.connect(self.load_model_and_captions)
        self.searchButton.clicked.connect(self.search_images)
        
    def browse_json(self):
        """Ouvre une boîte de dialogue pour sélectionner le fichier JSON"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Sélectionner le fichier JSON", "", "Fichiers JSON (*.json)"
        )
        if file_path:
            self.jsonEdit.setText(file_path)
    
    def browse_model(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier du modèle"""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Sélectionner le dossier du modèle"
        )
        if folder_path:
            self.modelEdit.setText(folder_path)
    
    def browse_embeddings(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier des embeddings"""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Sélectionner le dossier des embeddings"
        )
        if folder_path:
            self.embeddingsEdit.setText(folder_path)
    
    def load_model_and_captions(self):
        """Charge le modèle et les descriptions"""
        # Récupérer les chemins
        json_path = self.jsonEdit.text()
        model_path = self.modelEdit.text()
        
        # Vérifier que les fichiers existent
        if not os.path.exists(json_path):
            QtWidgets.QMessageBox.warning(
                self, 
                "Erreur", 
                f"Le fichier JSON {json_path} n'existe pas."
            )
            return
        
        # Mettre à jour la barre de progression
        self.progressBar.setValue(10)
        QtWidgets.QApplication.processEvents()
        
        try:
            # Charger les descriptions
            with open(json_path, 'r', encoding='utf-8') as f:
                self.captions = json.load(f)
            
            self.progressBar.setValue(50)
            QtWidgets.QApplication.processEvents()
            
            # Charger le modèle
            self.model = SentenceTransformer(model_path)
            
            self.progressBar.setValue(100)
            
            # Activer la recherche
            self.searchGroupBox.setEnabled(True)
            
            # Afficher un message de succès
            QtWidgets.QMessageBox.information(
                self, 
                "Succès", 
                f"Modèle et descriptions chargés avec succès.\n"
                f"Nombre de descriptions: {len(self.captions)}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 
                "Erreur", 
                f"Erreur lors du chargement: {str(e)}"
            )
            self.progressBar.setValue(0)
    
    def find_matching_race(self, race_name, races_list):
        """
        Trouve la race correspondante dans la liste, en tenant compte des différents formats
        """
        if not races_list:
            return None
        
        # Normaliser le nom de race du fichier
        race_name_lower = race_name.lower()
        
        # 1. Essayer une correspondance directe
        for race in races_list:
            if race.lower() == race_name_lower:
                return race
        
        # 2. Essayer en remplaçant les espaces par des underscores
        for race in races_list:
            if race.lower().replace(' ', '_') == race_name_lower:
                return race
        
        # 3. Essayer en supprimant les espaces
        for race in races_list:
            if race.lower().replace(' ', '') == race_name_lower:
                return race
        
        # 4. Essayer une correspondance partielle
        for race in races_list:
            race_lower = race.lower()
            if race_name_lower in race_lower or race_lower in race_name_lower:
                return race
            
            # Essayer aussi sans les espaces
            race_lower_no_spaces = race_lower.replace(' ', '')
            if race_name_lower in race_lower_no_spaces or race_lower_no_spaces in race_name_lower:
                return race
        
        # Si aucune correspondance n'est trouvée, retourner la première race de la liste
        return races_list[0]

    def find_image_path(self, image_filename, animal=None, race=None):
        """
        Trouve le chemin d'une image en utilisant directement les informations d'animal et de race
        """
        # Si animal et race sont fournis, utiliser directement ces informations
        if animal and race:
            # Essayer différentes extensions
            for img_ext in ['.jpg', '.jpeg', '.png']:
                # Extraire le nom de base sans extension
                base_name = os.path.splitext(image_filename)[0]
                direct_path = os.path.join('MIR_DATASETS_B', animal, race, f"{base_name}{img_ext}")
                if os.path.exists(direct_path):
                    return direct_path
            
            # Si aucune correspondance exacte, essayer de trouver la race correspondante
            races_list = None
            if animal == "araignee":
                races_list = self.araignees
            elif animal == "chiens":
                races_list = self.chiens
            elif animal == "oiseaux":
                races_list = self.oiseaux
            elif animal == "poissons":
                races_list = self.poissons
            elif animal == "singes":
                races_list = self.singes
            
            if races_list:
                matching_race = self.find_matching_race(race, races_list)
                if matching_race:
                    for img_ext in ['.jpg', '.jpeg', '.png']:
                        base_name = os.path.splitext(image_filename)[0]
                        direct_path = os.path.join('MIR_DATASETS_B', animal, matching_race, f"{base_name}{img_ext}")
                        if os.path.exists(direct_path):
                            return direct_path
        
        # Si on arrive ici ou si animal/race ne sont pas fournis, parcourir tous les animaux et races
        # Extraire le nom de base sans extension
        base_name = os.path.splitext(image_filename)[0]
        
        # Parcourir tous les animaux et races
        for animal in self.animaux:
            races_list = None
            if animal == "araignee":
                races_list = self.araignees
            elif animal == "chiens":
                races_list = self.chiens
            elif animal == "oiseaux":
                races_list = self.oiseaux
            elif animal == "poissons":
                races_list = self.poissons
            elif animal == "singes":
                races_list = self.singes
            
            if races_list:
                for race in races_list:
                    # Essayer de trouver l'image dans ce dossier
                    for img_ext in ['.jpg', '.jpeg', '.png']:
                        image_name = f"{base_name}{img_ext}"
                        direct_path = os.path.join('MIR_DATASETS_B', animal, race, image_name)
                        if os.path.exists(direct_path):
                            return direct_path
        
        # Si on arrive ici, on n'a pas trouvé le fichier
        return None

    def search_images(self):
        """Recherche les images correspondant à la description"""
        total_start_time = time.time()
        
        # Vérifier que le modèle est chargé
        if self.model is None:
            QtWidgets.QMessageBox.warning(
                self, 
                "Erreur", 
                "Veuillez d'abord charger le modèle."
            )
            return
        
        # Récupérer la requête
        query = self.searchEdit.text()
        if not query:
            QtWidgets.QMessageBox.warning(
                self, 
                "Erreur", 
                "Veuillez entrer une description."
            )
            return
        
        # Nombre de résultats
        top_k = self.resultsSpinBox.value()
        
        # Effacer les résultats précédents
        for i in reversed(range(self.scrollAreaLayout.count())):
            widget = self.scrollAreaLayout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # Afficher un message "Recherche en cours..."
        self.statusLabel = QtWidgets.QLabel("Recherche en cours...")
        self.statusLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.statusLabel.setStyleSheet("font-weight: bold; color: blue; font-size: 14px;")
        self.scrollAreaLayout.addWidget(self.statusLabel)
        
        # Désactiver le bouton de recherche pendant la recherche
        self.searchButton.setEnabled(False)
        self.searchButton.setText("Recherche en cours...")
        QtWidgets.QApplication.processEvents()  # Forcer la mise à jour de l'interface
        
        # Encoder la requête
        print("\n--- MESURE DES PERFORMANCES ---")
        encoding_start = time.time()
        query_embedding = self.model.encode(query)
        encoding_time = time.time() - encoding_start
        print(f"1. Encodage de la requête: {encoding_time:.4f} secondes")
        
        # Dossier des embeddings
        embeddings_dir = self.embeddingsEdit.text()
        
        # Recherche de l'image la plus proche
        results = []
        
        # Parcours des fichiers d'embeddings
        embedding_search_start = time.time()
        similarity_calc_total = 0
        image_path_search_total = 0
        file_count = 0
        
        for root, dirs, files in os.walk(embeddings_dir):
            for file in files:
                if file.endswith('_embedding.txt'):
                    file_count += 1
                    emb_path = os.path.join(root, file)
                    try:
                        # Charger l'embedding
                        emb = np.fromstring(open(emb_path).read(), sep=' ')
                        
                        # Calculer la similarité
                        sim_start = time.time()
                        sim = cosine_similarity([query_embedding], [emb])[0][0]
                        similarity_calc_total += time.time() - sim_start
                        
                        # Extraire le chemin relatif
                        relative_path = os.path.relpath(emb_path, embeddings_dir)
                        relative_path = relative_path.replace('_embedding.txt', '')
                        
                        # Extraire l'animal et la race
                        path_parts = relative_path.split('/')
                        animal = path_parts[0] if len(path_parts) > 0 else None
                        race = path_parts[1] if len(path_parts) > 1 else None
                        
                        # Construire le chemin de l'image
                        image_filename = os.path.basename(relative_path)
                        path_search_start = time.time()
                        image_path = self.find_image_path(image_filename, animal, race)
                        image_path_search_total += time.time() - path_search_start
                        
                        # Récupérer la description si disponible
                        caption = ""
                        for key in self.captions:
                            if image_filename in key:
                                caption = self.captions[key]
                                break
                        
                        # Ajouter aux résultats seulement si l'image est trouvée
                        if image_path:
                            results.append((image_path, caption, sim, animal, race))
                    except Exception as e:
                        print(f"Erreur lors du traitement de {emb_path}: {str(e)}")
        
        embedding_search_time = time.time() - embedding_search_start
        print(f"2. Parcours des embeddings ({file_count} fichiers): {embedding_search_time:.4f} secondes")
        print(f"3. Calcul de similarité (total): {similarity_calc_total:.4f} secondes")
        print(f"   Moyenne par image: {similarity_calc_total/max(1, file_count):.6f} secondes")
        print(f"4. Recherche des chemins d'images (total): {image_path_search_total:.4f} secondes")
        print(f"   Moyenne par image: {image_path_search_total/max(1, file_count):.6f} secondes")
        
        # Supprimer le message "Recherche en cours..."
        if hasattr(self, 'statusLabel'):
            self.statusLabel.deleteLater()
            delattr(self, 'statusLabel')
        
        # Trier les résultats par similarité décroissante
        sort_start = time.time()
        results.sort(key=lambda x: x[2], reverse=True)
        sort_time = time.time() - sort_start
        print(f"5. Tri des résultats: {sort_time:.4f} secondes")
        
        # Limiter aux top_k résultats
        results = results[:top_k]
        
        # Réactiver le bouton de recherche
        self.searchButton.setEnabled(True)
        self.searchButton.setText("Rechercher")
        
        # Afficher un message temporaire "Recherche terminée"
        self.statusLabel = QtWidgets.QLabel("Recherche terminée !")
        self.statusLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.statusLabel.setStyleSheet("font-weight: bold; color: green; font-size: 14px;")
        self.scrollAreaLayout.addWidget(self.statusLabel)
        QtWidgets.QApplication.processEvents()  # Forcer la mise à jour de l'interface
        
        # Programmer la suppression du message après 3 secondes
        QtCore.QTimer.singleShot(3000, lambda: self.statusLabel.deleteLater() if hasattr(self, 'statusLabel') else None)
        
        # Afficher les résultats
        display_start = time.time()
        self.display_results(results)
        display_time = time.time() - display_start
        print(f"6. Affichage des résultats: {display_time:.4f} secondes")
        
        total_time = time.time() - total_start_time
        print(f"Temps total de recherche: {total_time:.4f} secondes")
        print("--------------------------------\n")
    
    def display_results(self, results):
        """Affiche les résultats de la recherche"""
        # Effacer les résultats précédents
        for i in reversed(range(self.scrollAreaLayout.count())):
            widget = self.scrollAreaLayout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        if not results:
            # Aucun résultat
            label = QtWidgets.QLabel("Aucun résultat trouvé.")
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.scrollAreaLayout.addWidget(label)
            return
        
        # Afficher chaque résultat
        for i, (image_path, caption, similarity, animal, race) in enumerate(results):
            # Créer un widget pour ce résultat
            resultWidget = QtWidgets.QWidget()
            resultLayout = QtWidgets.QHBoxLayout(resultWidget)
            
            # Image
            imageLabel = QtWidgets.QLabel()
            imageLabel.setFixedSize(200, 200)
            imageLabel.setAlignment(QtCore.Qt.AlignCenter)
            
            # Essayer de charger l'image
            try:
                if os.path.exists(image_path):
                    pixmap = QtGui.QPixmap(image_path)
                    pixmap = pixmap.scaled(200, 200, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                    imageLabel.setPixmap(pixmap)
                else:
                    imageLabel.setText("Image non trouvée")
            except Exception as e:
                imageLabel.setText(f"Erreur: {str(e)}")
            
            resultLayout.addWidget(imageLabel)
            
            # Informations
            infoWidget = QtWidgets.QWidget()
            infoLayout = QtWidgets.QVBoxLayout(infoWidget)
            
            # Chemin de l'image
            pathLabel = QtWidgets.QLabel(f"<b>Chemin:</b> {image_path}")
            infoLayout.addWidget(pathLabel)
            
            # Description
            captionLabel = QtWidgets.QLabel(f"<b>Description:</b> {caption}")
            captionLabel.setWordWrap(True)
            infoLayout.addWidget(captionLabel)
            
            # Score de similarité
            scoreLabel = QtWidgets.QLabel(f"<b>Score de similarité:</b> {similarity:.4f}")
            infoLayout.addWidget(scoreLabel)
            
            # Animal et race
            animalLabel = QtWidgets.QLabel(f"<b>Animal:</b> {animal}")
            breedLabel = QtWidgets.QLabel(f"<b>Race:</b> {race}")
            infoLayout.addWidget(animalLabel)
            infoLayout.addWidget(breedLabel)
            
            infoLayout.addStretch()
            resultLayout.addWidget(infoWidget)
            
            # Ajouter ce résultat au layout principal
            self.scrollAreaLayout.addWidget(resultWidget)
            
            # Ajouter un séparateur sauf pour le dernier résultat
            if i < len(results) - 1:
                line = QtWidgets.QFrame()
                line.setFrameShape(QtWidgets.QFrame.HLine)
                line.setFrameShadow(QtWidgets.QFrame.Sunken)
                self.scrollAreaLayout.addWidget(line) 
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

class TextSearchPage(QtWidgets.QWidget):
    def __init__(self, parent=None, model_path="Transformer/sentence_transformer_model"):
        super(TextSearchPage, self).__init__(parent)
        self.model_path = model_path
        self.model = None
        self.captions = {}
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
    
    def search_images(self):
        """Recherche les images correspondant à la description"""
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
        
        # Encoder la requête
        query_embedding = self.model.encode(query)
        
        # Dossier des embeddings
        embeddings_dir = self.embeddingsEdit.text()
        
        # Recherche de l'image la plus proche
        results = []
        
        # Parcours des fichiers d'embeddings
        for root, dirs, files in os.walk(embeddings_dir):
            for file in files:
                if file.endswith('_embedding.txt'):
                    emb_path = os.path.join(root, file)
                    try:
                        # Charger l'embedding
                        emb = np.fromstring(open(emb_path).read(), sep=' ')
                        
                        # Calculer la similarité
                        sim = cosine_similarity([query_embedding], [emb])[0][0]
                        
                        # Extraire le chemin relatif
                        relative_path = os.path.relpath(emb_path, embeddings_dir)
                        relative_path = relative_path.replace('_embedding.txt', '')
                        
                        # Extraire l'animal et la race
                        path_parts = relative_path.split('/')
                        animal = path_parts[0] if len(path_parts) > 0 else "Inconnu"
                        race = path_parts[1] if len(path_parts) > 1 else "Inconnue"
                        
                        # Construire le chemin de l'image
                        image_filename = os.path.basename(relative_path)
                        image_path = find_image_in_directory('MIR_DATASETS_B', image_filename)
                        
                        # Récupérer la description si disponible
                        caption = ""
                        for key in self.captions:
                            if image_filename in key:
                                caption = self.captions[key]
                                break
                        
                        # Ajouter aux résultats
                        results.append((image_path, caption, sim, animal, race))
                    except Exception as e:
                        print(f"Erreur lors du traitement de {emb_path}: {str(e)}")
        
        # Trier les résultats par similarité décroissante
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Limiter aux top_k résultats
        results = results[:top_k]
        
        # Afficher les résultats
        self.display_results(results)
    
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
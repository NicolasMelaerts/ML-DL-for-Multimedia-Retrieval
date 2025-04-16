# -*- coding: utf-8 -*-
"""
Page d'affichage des images
"""

from PyQt5 import QtCore, QtGui, QtWidgets
import os

class DisplayPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DisplayPage, self).__init__(parent)
        self.setupUi()
        self.list_images = []
        self.current_index = -1
        
    def setupUi(self):
        self.setObjectName("DisplayPage")
        self.resize(1000, 600)
        
        # Layout principal
        self.mainLayout = QtWidgets.QVBoxLayout(self)
        
        # Titre
        self.titleLabel = QtWidgets.QLabel("Affichage des Images")
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.titleLabel.setFont(font)
        self.mainLayout.addWidget(self.titleLabel)
        
        # Espace
        self.mainLayout.addSpacing(20)
        
        # Bouton pour charger un dossier d'images
        self.loadButton = QtWidgets.QPushButton("Charger un dossier d'images")
        self.loadButton.setMinimumHeight(40)
        self.mainLayout.addWidget(self.loadButton)
        
        # Espace
        self.mainLayout.addSpacing(10)
        
        # Zone d'affichage de l'image
        self.imageFrame = QtWidgets.QFrame()
        self.imageFrame.setFrameShape(QtWidgets.QFrame.Box)
        self.imageFrame.setMinimumSize(600, 400)
        
        # Layout pour l'image
        self.imageLayout = QtWidgets.QVBoxLayout(self.imageFrame)
        
        # Label pour l'image
        self.image = QtWidgets.QLabel()
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.setText("Aucune image chargée")
        self.imageLayout.addWidget(self.image)
        
        # Ajouter le cadre de l'image au layout principal
        self.mainLayout.addWidget(self.imageFrame)
        
        # Layout pour les contrôles de navigation
        self.navigationLayout = QtWidgets.QHBoxLayout()
        
        # Bouton précédent
        self.prevButton = QtWidgets.QPushButton("◀ Précédente")
        self.prevButton.setMinimumHeight(30)
        self.prevButton.setEnabled(False)
        self.navigationLayout.addWidget(self.prevButton)
        
        # Affichage du numéro d'image
        self.imageCountLabel = QtWidgets.QLabel("0 / 0")
        self.imageCountLabel.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setBold(True)
        self.imageCountLabel.setFont(font)
        self.navigationLayout.addWidget(self.imageCountLabel)
        
        # Bouton suivant
        self.nextButton = QtWidgets.QPushButton("Suivante ▶")
        self.nextButton.setMinimumHeight(30)
        self.nextButton.setEnabled(False)
        self.navigationLayout.addWidget(self.nextButton)
        
        # Ajouter les contrôles de navigation au layout principal
        self.mainLayout.addLayout(self.navigationLayout)
        
        # Espace
        self.mainLayout.addSpacing(10)
        
        # Layout pour la liste déroulante
        self.comboLayout = QtWidgets.QHBoxLayout()
        
        # Label pour la liste
        self.comboLabel = QtWidgets.QLabel("Sélectionner une image:")
        self.comboLayout.addWidget(self.comboLabel)
        
        # Liste déroulante des images
        self.imageComboBox = QtWidgets.QComboBox()
        self.imageComboBox.setMinimumWidth(400)
        self.comboLayout.addWidget(self.imageComboBox)
        
        # Ajouter la liste déroulante au layout principal
        self.mainLayout.addLayout(self.comboLayout)
        
        # Espace
        self.mainLayout.addSpacing(20)
        
        # Bouton retour
        self.backButton = QtWidgets.QPushButton("Retour à l'accueil")
        self.backButton.setMinimumHeight(40)
        self.mainLayout.addWidget(self.backButton)
        
        # Connexion des signaux
        self.loadButton.clicked.connect(self.loadImages)
        self.prevButton.clicked.connect(self.showPreviousImage)
        self.nextButton.clicked.connect(self.showNextImage)
        self.imageComboBox.currentIndexChanged.connect(self.onComboBoxChanged)
    
    def loadImages(self):
        self.Dossier_images = QtWidgets.QFileDialog.getExistingDirectory(
            None, 'Sélectionner un dossier d\'images', "", QtWidgets.QFileDialog.ShowDirsOnly
        )
        
        if not self.Dossier_images:
            return
            
        # Liste pour stocker les chemins des images
        self.list_images = []
        
        # Parcourir le dossier pour trouver les images
        for filename in os.listdir(self.Dossier_images):
            file_path = os.path.join(self.Dossier_images, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.list_images.append(file_path)
        
        if not self.list_images:
            QtWidgets.QMessageBox.warning(
                self, 
                "Aucune image", 
                "Aucune image trouvée dans le dossier sélectionné"
            )
            return
        
        # Trier les images par nom
        self.list_images.sort()
        
        # Remplir la liste déroulante
        self.imageComboBox.clear()
        for file_path in self.list_images:
            self.imageComboBox.addItem(os.path.basename(file_path), file_path)
        
        # Activer les boutons de navigation
        self.updateNavigationButtons()
        
        # Afficher la première image
        self.current_index = 0
        self.displayCurrentImage()
    
    def displayCurrentImage(self):
        if 0 <= self.current_index < len(self.list_images):
            file_path = self.list_images[self.current_index]
            
            # Mettre à jour la liste déroulante
            self.imageComboBox.setCurrentIndex(self.current_index)
            
            # Mettre à jour le label de comptage
            self.imageCountLabel.setText(f"{self.current_index + 1} / {len(self.list_images)}")
            
            # Afficher l'image
            pixmap = QtGui.QPixmap(file_path)
            pixmap = pixmap.scaled(self.image.width(), self.image.height(), 
                                  QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image.setPixmap(pixmap)
            
            # Mettre à jour les boutons de navigation
            self.updateNavigationButtons()
    
    def showPreviousImage(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.displayCurrentImage()
    
    def showNextImage(self):
        if self.current_index < len(self.list_images) - 1:
            self.current_index += 1
            self.displayCurrentImage()
    
    def onComboBoxChanged(self, index):
        if index != self.current_index and index >= 0:
            self.current_index = index
            self.displayCurrentImage()
    
    def updateNavigationButtons(self):
        # Activer/désactiver les boutons de navigation selon la position actuelle
        self.prevButton.setEnabled(self.current_index > 0)
        self.nextButton.setEnabled(self.current_index < len(self.list_images) - 1) 
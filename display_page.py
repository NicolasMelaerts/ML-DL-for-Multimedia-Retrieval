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
        
        # Layout pour l'affichage des images et la liste
        self.displayLayout = QtWidgets.QHBoxLayout()
        
        # Zone d'affichage de l'image
        self.imageFrame = QtWidgets.QFrame()
        self.imageFrame.setFrameShape(QtWidgets.QFrame.Box)
        self.imageFrame.setMinimumSize(500, 400)
        
        # Layout pour l'image
        self.imageLayout = QtWidgets.QVBoxLayout(self.imageFrame)
        
        # Label pour l'image
        self.image = QtWidgets.QLabel()
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.setText("Aucune image chargée")
        self.imageLayout.addWidget(self.image)
        
        # Ajouter le cadre de l'image au layout d'affichage
        self.displayLayout.addWidget(self.imageFrame)
        
        # Liste des images
        self.tableView = QtWidgets.QTableView()
        self.tableView.setMinimumWidth(300)
        self.tableView.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableView.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.displayLayout.addWidget(self.tableView)
        
        # Ajouter le layout d'affichage au layout principal
        self.mainLayout.addLayout(self.displayLayout)
        
        # Espace
        self.mainLayout.addSpacing(20)
        
        # Bouton retour
        self.backButton = QtWidgets.QPushButton("Retour à l'accueil")
        self.backButton.setMinimumHeight(40)
        self.mainLayout.addWidget(self.backButton)
        
        # Connexion des signaux
        self.loadButton.clicked.connect(self.loadImages)
    
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
            
        # Afficher la première image
        pixmap = QtGui.QPixmap(self.list_images[0])
        pixmap = pixmap.scaled(self.image.width(), self.image.height(), 
                              QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image.setPixmap(pixmap)
        
        # Créer le modèle pour le TableView
        model = QtGui.QStandardItemModel()
        model.setHorizontalHeaderLabels(["Nom du fichier"])
        
        # Ajouter les images à la liste
        for file_path in self.list_images:
            item = QtGui.QStandardItem(os.path.basename(file_path))
            item.setData(file_path, QtCore.Qt.UserRole + 1)
            item.setEditable(False)
            model.appendRow(item)
        
        self.tableView.setModel(model)
        
        # Connecter le signal de sélection
        self.tableView.selectionModel().currentChanged.connect(self.displayImage)
    
    def displayImage(self, current, previous):
        if current.isValid():
            file_path = current.data(QtCore.Qt.UserRole + 1)
            if not file_path:
                file_path = self.list_images[current.row()]
                
            pixmap = QtGui.QPixmap(file_path)
            pixmap = pixmap.scaled(self.image.width(), self.image.height(), 
                                  QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image.setPixmap(pixmap) 
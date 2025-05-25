# -*- coding: utf-8 -*-
"""
Page d'accueil de l'application
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from descriptors_page import DescriptorsPage
from display_page import DisplayPage
from text_search_page import TextSearchPage
from deep_search_page import DeepSearchPage
from search_page import SearchPage
import os

class FeatureCard(QtWidgets.QFrame):
    def __init__(self, title, description, button_text, button_color, parent=None):
        super(FeatureCard, self).__init__(parent)
        
        # Style de la carte
        self.setObjectName("FeatureCard")
        self.setStyleSheet("""
            #FeatureCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        # Layout principal de la carte
        self.cardLayout = QtWidgets.QVBoxLayout(self)
        self.cardLayout.setContentsMargins(15, 15, 15, 15)
        self.cardLayout.setSpacing(10)
        
        # Titre
        self.titleLabel = QtWidgets.QLabel(title)
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.titleLabel.setFont(font)
        self.titleLabel.setStyleSheet(f"color: {button_color};")
        self.cardLayout.addWidget(self.titleLabel)
        
        # Description
        self.descLabel = QtWidgets.QLabel(description)
        self.descLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.descLabel.setWordWrap(True)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.descLabel.setFont(font)
        self.cardLayout.addWidget(self.descLabel)
        
        # Espace
        self.cardLayout.addStretch()
        
        # Bouton
        self.button = QtWidgets.QPushButton(button_text)
        self.button.setMinimumHeight(35)
        self.button.setStyleSheet(f"""
            QPushButton {{
                background-color: {button_color};
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {self.darken_color(button_color)};
            }}
        """)
        self.cardLayout.addWidget(self.button)
    
    def darken_color(self, hex_color, factor=0.8):
        """Assombrir une couleur hexadécimale"""
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
        rgb = tuple(int(c * factor) for c in rgb)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

class HomePage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(HomePage, self).__init__(parent)
        self.setupUi()
        
    def setupUi(self):
        self.setObjectName("HomePage")
        self.resize(1000, 700)
        
        # Layout principal
        self.mainLayout = QtWidgets.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(20, 20, 20, 20)
        self.mainLayout.setSpacing(15)
        
        # En-tête sans cadre bleu
        self.headerLayout = QtWidgets.QVBoxLayout()
        self.headerLayout.setContentsMargins(20, 20, 20, 20)
        self.headerLayout.setSpacing(10)
        
        # Titre
        self.titleLabel = QtWidgets.QLabel("Système de Recherche d'Images")
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.titleLabel.setFont(font)
        self.headerLayout.addWidget(self.titleLabel)
        
        # Sous-titre
        self.subtitleLabel = QtWidgets.QLabel("Plateforme avancée de recherche et d'analyse d'images")
        self.subtitleLabel.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.subtitleLabel.setFont(font)
        self.headerLayout.addWidget(self.subtitleLabel)
        
        # Séparateur
        self.separator = QtWidgets.QFrame()
        self.separator.setFrameShape(QtWidgets.QFrame.HLine)
        self.separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.headerLayout.addWidget(self.separator)
        
        # Description
        self.descLabel = QtWidgets.QLabel("Explorez notre base d'images, calculez des descripteurs et utilisez différentes méthodes de recherche.")
        self.descLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.descLabel.setWordWrap(True)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.descLabel.setFont(font)
        self.headerLayout.addWidget(self.descLabel)
        
        # Ajouter l'en-tête au layout principal
        self.mainLayout.addLayout(self.headerLayout)
        
        # Créer la première rangée (Affichage et Descripteurs)
        self.topRowLayout = QtWidgets.QHBoxLayout()
        self.topRowLayout.setSpacing(15)
        
        # Affichage des Images
        self.displayCard = FeatureCard(
            "Affichage des Images",
            "Parcourez et visualisez les images disponibles dans la base de données.",
            "Explorer",
            "#3498db"
        )
        self.displayCard.button.clicked.connect(self.openDisplayPage)
        self.topRowLayout.addWidget(self.displayCard)
        
        # Calcul des Descripteurs
        self.descriptorsCard = FeatureCard(
            "Calcul des Descripteurs",
            "Calculez différents descripteurs pour indexer votre base d'images.",
            "Configurer",
            "#6c757d"
        )
        self.descriptorsCard.button.clicked.connect(self.openDescriptorsPage)
        self.topRowLayout.addWidget(self.descriptorsCard)
        
        # Ajouter la première rangée au layout principal
        self.mainLayout.addLayout(self.topRowLayout)
        
        # Créer la deuxième rangée (3 méthodes de recherche)
        self.bottomRowLayout = QtWidgets.QHBoxLayout()
        self.bottomRowLayout.setSpacing(15)
        
        # Recherche par Descripteurs
        self.searchCard = FeatureCard(
            "Recherche par Descripteurs",
            "Recherchez des images similaires en utilisant les descripteurs calculés.",
            "Rechercher",
            "#28a745"
        )
        self.searchCard.button.clicked.connect(self.openSearchPage)
        self.bottomRowLayout.addWidget(self.searchCard)
        
        # Recherche par Texte
        self.textSearchCard = FeatureCard(
            "Recherche par Texte",
            "Trouvez des images en utilisant des descriptions textuelles.",
            "Rechercher",
            "#17a2b8"
        )
        self.textSearchCard.button.clicked.connect(self.openTextSearchPage)
        self.bottomRowLayout.addWidget(self.textSearchCard)
        
        # Recherche Deep Learning
        self.deepSearchCard = FeatureCard(
            "Recherche Deep Learning",
            "Utilisez des modèles de deep learning pour trouver des images similaires.",
            "Rechercher",
            "#ffc107"
        )
        self.deepSearchCard.button.clicked.connect(self.openDeepSearchPage)
        self.bottomRowLayout.addWidget(self.deepSearchCard)
        
        # Ajouter la deuxième rangée au layout principal
        self.mainLayout.addLayout(self.bottomRowLayout)
        
        # Ajouter un espace extensible en bas
        self.mainLayout.addStretch()
    
    def openDescriptorsPage(self):
        self.descriptorsPage = DescriptorsPage()
        self.descriptorsPage.backButton.clicked.connect(self.showHomePage)
        self.hide()
        self.descriptorsPage.show()
    
    def openSearchPage(self):
        self.searchPage = SearchPage()
        self.searchPage.backButton.clicked.connect(self.showHomePage)
        self.hide()
        self.searchPage.show()
    
    def openTextSearchPage(self):
        self.textSearchPage = TextSearchPage()
        self.textSearchPage.backButton.clicked.connect(self.showHomePage)
        self.hide()
        self.textSearchPage.show()
    
    def openDisplayPage(self):
        self.displayPage = DisplayPage()
        self.displayPage.backButton.clicked.connect(self.showHomePage)
        self.hide()
        self.displayPage.show()
    
    def openDeepSearchPage(self):
        self.deepSearchPage = DeepSearchPage()
        self.deepSearchPage.backButton.clicked.connect(self.showHomePage)
        self.hide()
        self.deepSearchPage.show()
    
    def showHomePage(self):
        sender = self.sender()
        sender.parent().hide()
        self.show()
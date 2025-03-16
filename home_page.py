# -*- coding: utf-8 -*-
"""
Page d'accueil de l'application
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from descriptors_page import DescriptorsPage
from display_page import DisplayPage

class HomePage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(HomePage, self).__init__(parent)
        self.setupUi()
        
    def setupUi(self):
        self.setObjectName("HomePage")
        self.resize(800, 600)
        
        # Layout principal
        self.mainLayout = QtWidgets.QVBoxLayout(self)
        
        # Titre
        self.titleLabel = QtWidgets.QLabel("Système de Recherche d'Images")
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.titleLabel.setFont(font)
        self.mainLayout.addWidget(self.titleLabel)
        
        # Espace
        self.mainLayout.addSpacing(40)
        
        # Boutons
        self.btnLayout = QtWidgets.QVBoxLayout()
        
        # Bouton pour le calcul des descripteurs
        self.btnDescriptors = QtWidgets.QPushButton("Calcul des Descripteurs")
        self.btnDescriptors.setMinimumHeight(60)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btnDescriptors.setFont(font)
        self.btnLayout.addWidget(self.btnDescriptors)
        
        # Espace entre les boutons
        self.btnLayout.addSpacing(20)
        
        # Bouton pour le moteur de recherche
        self.btnSearch = QtWidgets.QPushButton("Moteur de Recherche")
        self.btnSearch.setMinimumHeight(60)
        self.btnSearch.setFont(font)
        self.btnLayout.addWidget(self.btnSearch)
        
        # Espace entre les boutons
        self.btnLayout.addSpacing(20)
        
        # Bouton pour l'affichage des images
        self.btnDisplay = QtWidgets.QPushButton("Affichage des Images")
        self.btnDisplay.setMinimumHeight(60)
        self.btnDisplay.setFont(font)
        self.btnLayout.addWidget(self.btnDisplay)
        
        # Ajouter les boutons au layout principal avec des marges
        self.mainLayout.addLayout(self.btnLayout)
        self.mainLayout.addStretch(1)
        
        # Connexion des signaux
        self.btnDescriptors.clicked.connect(self.openDescriptorsPage)
        self.btnSearch.clicked.connect(self.openSearchPage)
        self.btnDisplay.clicked.connect(self.openDisplayPage)
    
    def openDescriptorsPage(self):
        self.descriptorsPage = DescriptorsPage()
        self.descriptorsPage.backButton.clicked.connect(self.showHomePage)
        self.hide()
        self.descriptorsPage.show()
    
    def openSearchPage(self):
        from search_page import SearchPage
        self.searchPage = SearchPage()
        self.searchPage.backButton.clicked.connect(self.showHomePage)
        self.hide()
        self.searchPage.show()
    
    def openDisplayPage(self):
        self.displayPage = DisplayPage()
        self.displayPage.backButton.clicked.connect(self.showHomePage)
        self.hide()
        self.displayPage.show()
    
    def showHomePage(self):
        sender = self.sender()
        sender.parent().hide()
        self.show() 
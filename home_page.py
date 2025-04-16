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
        
        # Bouton pour l'affichage des images (une seule fois)
        self.btnDisplay = QtWidgets.QPushButton("Affichage des Images")
        self.btnDisplay.setMinimumHeight(60)
        self.btnDisplay.setFont(font)
        self.btnLayout.addWidget(self.btnDisplay)

        # Espace entre les boutons
        self.btnLayout.addSpacing(20)
        
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
        self.btnSearch = QtWidgets.QPushButton("Moteur de Recherche avec descripteurs")
        self.btnSearch.setMinimumHeight(60)
        self.btnSearch.setFont(font)
        self.btnLayout.addWidget(self.btnSearch)
        
        # Espace entre les boutons
        self.btnLayout.addSpacing(20)
        
        # Bouton pour la recherche d'images par texte
        self.btnTextSearch = QtWidgets.QPushButton("Recherche d'Images par Texte")
        self.btnTextSearch.setMinimumHeight(60)
        self.btnTextSearch.setFont(font)
        self.btnLayout.addWidget(self.btnTextSearch)
        
        # Espace entre les boutons
        self.btnLayout.addSpacing(20)
        
        # Bouton pour la recherche par deep learning
        self.btnDeepSearch = QtWidgets.QPushButton("Moteur de Recherche par Deep Learning")
        self.btnDeepSearch.setMinimumHeight(60)
        self.btnDeepSearch.setFont(font)
        self.btnLayout.addWidget(self.btnDeepSearch)
        
        # Ajouter les boutons au layout principal avec des marges
        self.mainLayout.addLayout(self.btnLayout)
        self.mainLayout.addStretch(1)
        
        # Connexion des signaux
        self.btnDescriptors.clicked.connect(self.openDescriptorsPage)
        self.btnSearch.clicked.connect(self.openSearchPage)
        self.btnTextSearch.clicked.connect(self.openTextSearchPage)
        self.btnDisplay.clicked.connect(self.openDisplayPage)
        self.btnDeepSearch.clicked.connect(self.openDeepSearchPage)
    
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
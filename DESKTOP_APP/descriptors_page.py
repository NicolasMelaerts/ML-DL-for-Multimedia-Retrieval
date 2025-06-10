# -*- coding: utf-8 -*-
"""
Page de calcul des descripteurs
"""

from PyQt5 import QtCore, QtGui, QtWidgets
import os
from descriptors import (
    generateHistogramme_Color, 
    generateHistogramme_HSV, 
    generateORB,
    generateGLCM,
    generateLBP,
    generateHOG
)

def showDialog():
    msgBox = QtWidgets.QMessageBox()
    msgBox.setIcon(QtWidgets.QMessageBox.Information)
    msgBox.setText("Merci de sélectionner un descripteur via le menu ci-dessus")
    msgBox.setWindowTitle("Pas de Descripteur sélectionné")
    msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
    returnValue = msgBox.exec()

class DescriptorsPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DescriptorsPage, self).__init__(parent)
        self.setupUi()
        self.Dossier_images = None
        
    def setupUi(self):
        self.setObjectName("DescriptorsPage")
        self.resize(1000, 600)
        
        # Définir le titre de la fenêtre
        self.setWindowTitle("Calcul des Descripteurs")
        
        # Layout principal
        self.mainLayout = QtWidgets.QVBoxLayout(self)
        
        # Espace
        self.mainLayout.addSpacing(20)
        
        # Affichage du statut de la base de données
        self.dbStatusLabel = QtWidgets.QLabel("Statut: Aucune base de données chargée")
        self.dbStatusLabel.setStyleSheet("color: red; font-weight: bold;")
        self.mainLayout.addWidget(self.dbStatusLabel)
        
        # Layout pour les descripteurs
        self.descriptorsLayout = QtWidgets.QGridLayout()
        
        # Dictionnaire des descripteurs avec leurs descriptions
        self.descriptors_info = {
            "ORB": "Oriented FAST and Rotated BRIEF (ORB)\n\n"
                   "• Alternative rapide à SIFT et SURF\n"
                   "• Combine le détecteur FAST et le descripteur BRIEF\n"
                   "• Gère l'orientation pour plus de robustesse\n"
                   "• Efficace en termes de calcul",
                   
            "Hist Couleur ": "Histogramme de Couleur BGR\n\n"
                           "• Capture la distribution des couleurs (bleu, vert, rouge)\n"
                           "• Simple mais efficace pour décrire le contenu global\n"
                           "• Sensible aux changements d'illumination\n"
                           "• Ignore la disposition spatiale des couleurs",
                           
            "Hist HSV": "Histogramme HSV\n\n"
                        "• Représente les couleurs en Teinte-Saturation-Valeur\n"
                        "• Plus proche de la perception humaine des couleurs\n"
                        "• Meilleure robustesse aux changements d'illumination\n"
                        "• Utile pour la segmentation basée sur la couleur",
                        
            "GLCM": "Matrice de Co-occurrence des Niveaux de Gris (GLCM)\n\n"
                    "• Analyse la texture en considérant les relations spatiales\n"
                    "• Calcule la fréquence d'apparition de paires de pixels\n"
                    "• Extrait des propriétés comme contraste, homogénéité, énergie\n"
                    "• Efficace pour distinguer différentes textures",
                    
            "LBP": "Motif Binaire Local (LBP)\n\n"
                   "• Descripteur de texture robuste\n"
                   "• Étiquette les pixels en seuillant leur voisinage\n"
                   "• Invariant aux changements monotones d'illumination\n"
                   "• Utilisé pour la reconnaissance faciale et l'analyse de texture",
                   
            "HOG": "Histogramme des Gradients Orientés (HOG)\n\n"
                   "• Compte les occurrences des orientations de gradient\n"
                   "• Divise l'image en cellules et blocs\n"
                   "• Robuste aux changements d'illumination et aux petites déformations\n"
                   "• Très utilisé pour la détection d'objets et de personnes"
        }
        
        # Créer les checkboxes avec tooltips pour chaque descripteur
        row, col = 0, 0
        for desc_name, desc_info in self.descriptors_info.items():
            # Créer la checkbox
            checkbox = QtWidgets.QCheckBox(desc_name)
            checkbox.setToolTip(desc_info)  # Ajouter le tooltip formaté
            
            # Ajouter la checkbox au grid layout
            self.descriptorsLayout.addWidget(checkbox, row, col)
            
            # Passer à la colonne suivante ou à la ligne suivante si nécessaire
            col += 1
            if col > 3:  # 4 colonnes maximum
                col = 0
                row += 1
            
            # Stocker la référence à la checkbox
            attr_name = f"checkBox_{desc_name.replace(' ', '')}"
            setattr(self, attr_name, checkbox)
        
        self.mainLayout.addLayout(self.descriptorsLayout)
        
        # Espace
        self.mainLayout.addSpacing(20)
        
        # Boutons
        self.buttonLayout = QtWidgets.QHBoxLayout()
        
        # Bouton pour charger la base de données
        self.charger = QtWidgets.QPushButton("Charger la base de données")
        self.charger.setMinimumHeight(50)
        self.buttonLayout.addWidget(self.charger)
        
        # Bouton pour calculer les descripteurs
        self.indexer = QtWidgets.QPushButton("Calculer les descripteurs")
        self.indexer.setMinimumHeight(50)
        self.buttonLayout.addWidget(self.indexer)
        
        self.mainLayout.addLayout(self.buttonLayout)
        
        # Espace
        self.mainLayout.addSpacing(20)
        
        # Label pour afficher le descripteur en cours de calcul
        self.currentDescriptorLabel = QtWidgets.QLabel("Descripteur en cours: Aucun")
        self.currentDescriptorLabel.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setBold(True)
        self.currentDescriptorLabel.setFont(font)
        self.mainLayout.addWidget(self.currentDescriptorLabel)
        
        # Barre de progression
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setValue(0)
        self.mainLayout.addWidget(self.progressBar)
        
        # Zone de log
        self.logTextEdit = QtWidgets.QTextEdit()
        self.logTextEdit.setReadOnly(True)
        self.logTextEdit.setMaximumHeight(150)
        self.mainLayout.addWidget(self.logTextEdit)
        
        # Espace
        self.mainLayout.addSpacing(20)
        
        # Bouton retour
        self.backButton = QtWidgets.QPushButton("Retour à l'accueil")
        self.backButton.setMinimumHeight(40)
        self.backButton.setProperty("class", "home-button")
        self.mainLayout.addWidget(self.backButton)
        
        # Connexion des signaux
        self.charger.clicked.connect(self.loadDatabase)
        self.indexer.clicked.connect(self.extractFeatures)
    
    def loadDatabase(self):
        self.Dossier_images = QtWidgets.QFileDialog.getExistingDirectory(
            None, 'Sélectionner la base de données', "", QtWidgets.QFileDialog.ShowDirsOnly
        )
        
        if self.Dossier_images:
            self.dbStatusLabel.setText(f"Statut: Base de données chargée depuis: {self.Dossier_images}")
            self.dbStatusLabel.setStyleSheet("color: green; font-weight: bold;")
            self.logTextEdit.append(f"Base de données chargée depuis: {self.Dossier_images}")
            
            # Vérifier quels descripteurs sont déjà calculés
            self.updateDescriptorStatus()
    
    def updateDescriptorStatus(self):
        """Vérifie quels descripteurs sont déjà calculés et met à jour l'interface"""
        if not self.Dossier_images:
            return
            
        descriptors = {
            "BGR": self.checkBox_HistCouleur,
            "HSV": self.checkBox_HistHSV,
            "ORB": self.checkBox_ORB,
            "GLCM": self.checkBox_GLCM,
            "LBP": self.checkBox_LBP,
            "HOG": self.checkBox_HOG
        }
        
        # Vérifier si le dossier Descripteurs existe
        if not os.path.exists("Descripteurs"):
            return
        
        for desc_name, checkbox in descriptors.items():
            # Vérifier si le sous-dossier du descripteur existe dans le dossier Descripteurs
            desc_path = os.path.join("Descripteurs", desc_name)
            if os.path.exists(desc_path):
                checkbox.setStyleSheet("color: green;")
                checkbox.setToolTip(f"Le descripteur {desc_name} est déjà calculé")
                self.logTextEdit.append(f"Descripteur {desc_name} déjà calculé")
            else:
                checkbox.setStyleSheet("")
                checkbox.setToolTip(self.descriptors_info.get(desc_name, ""))
    
    # Ajouter une méthode pour mettre à jour la barre de progression
    def update_progress(self, value):
        self.progressBar.setValue(value)
    
    def extractFeatures(self):
        if not self.Dossier_images:
            QtWidgets.QMessageBox.warning(
                self, 
                "Erreur", 
                "Merci de charger la base de données d'abord"
            )
            return

        # Vérifier si l'utilisateur a sélectionné un descripteur
        if not (self.checkBox_ORB.isChecked() or self.checkBox_HistCouleur.isChecked() or 
                self.checkBox_HistHSV.isChecked() or
                self.checkBox_GLCM.isChecked() or self.checkBox_LBP.isChecked() or
                self.checkBox_HOG.isChecked()):
            showDialog()
            return

        # Liste pour suivre les descripteurs calculés
        calculated_descriptors = []
        
        # Utilisation des fonctions de `descriptors.py` avec notre méthode de callback
        if self.checkBox_HistCouleur.isChecked():
            desc_path = os.path.join("Descripteurs", "BGR")
            if os.path.exists(desc_path):
                self.logTextEdit.append("Descripteur BGR déjà calculé, calcul ignoré")
            else:
                self.currentDescriptorLabel.setText("Descripteur en cours: Histogramme Couleur")
                self.logTextEdit.append("Calcul du descripteur Histogramme Couleur...")
                generateHistogramme_Color(self.Dossier_images, self.update_progress)
                calculated_descriptors.append("Histogramme Couleur")
        
        if self.checkBox_HistHSV.isChecked():
            desc_path = os.path.join("Descripteurs", "HSV")
            if os.path.exists(desc_path):
                self.logTextEdit.append("Descripteur HSV déjà calculé, calcul ignoré")
            else:
                self.currentDescriptorLabel.setText("Descripteur en cours: Histogramme HSV")
                self.logTextEdit.append("Calcul du descripteur Histogramme HSV...")
                generateHistogramme_HSV(self.Dossier_images, self.update_progress)
                calculated_descriptors.append("Histogramme HSV")
        
        if self.checkBox_ORB.isChecked():
            desc_path = os.path.join("Descripteurs", "ORB")
            if os.path.exists(desc_path):
                self.logTextEdit.append("Descripteur ORB déjà calculé, calcul ignoré")
            else:
                self.currentDescriptorLabel.setText("Descripteur en cours: ORB")
                self.logTextEdit.append("Calcul du descripteur ORB...")
                generateORB(self.Dossier_images, self.update_progress)
                calculated_descriptors.append("ORB")
        
        if self.checkBox_GLCM.isChecked():
            desc_path = os.path.join("Descripteurs", "GLCM")
            if os.path.exists(desc_path):
                self.logTextEdit.append("Descripteur GLCM déjà calculé, calcul ignoré")
            else:
                self.currentDescriptorLabel.setText("Descripteur en cours: GLCM")
                self.logTextEdit.append("Calcul du descripteur GLCM...")
                generateGLCM(self.Dossier_images, self.update_progress)
                calculated_descriptors.append("GLCM")
        
        if self.checkBox_LBP.isChecked():
            desc_path = os.path.join("Descripteurs", "LBP")
            if os.path.exists(desc_path):
                self.logTextEdit.append("Descripteur LBP déjà calculé, calcul ignoré")
            else:
                self.currentDescriptorLabel.setText("Descripteur en cours: LBP")
                self.logTextEdit.append("Calcul du descripteur LBP...")
                generateLBP(self.Dossier_images, self.update_progress)
                calculated_descriptors.append("LBP")
        
        if self.checkBox_HOG.isChecked():
            desc_path = os.path.join("Descripteurs", "HOG")
            if os.path.exists(desc_path):
                self.logTextEdit.append("Descripteur HOG déjà calculé, calcul ignoré")
            else:
                self.currentDescriptorLabel.setText("Descripteur en cours: HOG")
                self.logTextEdit.append("Calcul du descripteur HOG...")
                generateHOG(self.Dossier_images, self.update_progress)
                calculated_descriptors.append("HOG")

        # Réinitialiser le label du descripteur en cours
        self.currentDescriptorLabel.setText("Descripteur en cours: Aucun")
        
        # Mettre à jour le statut des descripteurs
        self.updateDescriptorStatus()
        
        # Afficher un message de succès
        if calculated_descriptors:
            QtWidgets.QMessageBox.information(
                self, 
                "Extraction terminée", 
                f"Les descripteurs suivants ont été calculés avec succès:\n{', '.join(calculated_descriptors)}"
            )
        else:
            QtWidgets.QMessageBox.information(
                self, 
                "Information", 
                "Tous les descripteurs sélectionnés étaient déjà calculés."
            ) 

    def calculateDescriptors(self):
        """Calcule les descripteurs pour toutes les images"""
        
        # Déterminer quels descripteurs calculer
        descriptors_to_calculate = []
        if self.checkBoxColor.isChecked():
            descriptors_to_calculate.append(('BGR', 1))
        if self.checkBoxHOG.isChecked():
            descriptors_to_calculate.append(('HOG', 2))
        if self.checkBoxLBP.isChecked():
            descriptors_to_calculate.append(('LBP', 3))
        if self.checkBoxORB.isChecked():
            descriptors_to_calculate.append(('ORB', 4))
        if self.checkBoxHSV.isChecked():
            descriptors_to_calculate.append(('HSV', 5))
        if self.checkBoxGLCM.isChecked():
            descriptors_to_calculate.append(('GLCM', 6))
        
        
        print(f"Descripteurs à calculer: {[desc[0] for desc in descriptors_to_calculate]}")
    

    def checkAvailableDescriptors(self):
        """Vérifie quels descripteurs sont disponibles et met à jour l'interface"""
        
        # Liste des descripteurs et leurs checkboxes correspondantes (sans SIFT)
        descriptors_checkboxes = {
            'BGR': self.checkBoxColor,
            'HSV': self.checkBoxHSV,
            'GLCM': self.checkBoxGLCM,
            'HOG': self.checkBoxHOG,
            'LBP': self.checkBoxLBP,
            'ORB': self.checkBoxORB
        }
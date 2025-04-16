# -*- coding: utf-8 -*-
"""
Point d'entrée principal de l'application
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from home_page import HomePage  # Importez votre classe de page d'accueil

def main():
    # Créer l'application
    app = QApplication(sys.argv)
    
    # Charger et appliquer le style QSS
    style_file = os.path.join(os.path.dirname(__file__), "style.qss")
    if os.path.exists(style_file):
        with open(style_file, "r") as f:
            style = f.read()
            app.setStyleSheet(style)
            print("Style QSS appliqué avec succès")
    else:
        print(f"Fichier de style non trouvé: {style_file}")
    
    # Créer et afficher la fenêtre principale
    window = HomePage()
    window.setWindowTitle("Moteur de recherche par Nicolas Melaerts et Kenza Khemar")
    window.show()
    
    # Exécuter l'application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 
# -*- coding: utf-8 -*-
"""
Point d'entrée principal de l'application
"""

from PyQt5 import QtWidgets
from home_page import HomePage

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    
    # Créer et afficher la page d'accueil
    homePage = HomePage()
    homePage.show()
    
    sys.exit(app.exec_()) 
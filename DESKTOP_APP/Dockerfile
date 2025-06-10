# Utiliser une image de base avec PyQt et support X11
FROM coolsa/pyqt-designer:x64

# Définir le dossier de travail
WORKDIR /opt/DESKTOP_APP

# Installer les dépendances Python nécessaires
RUN pip install --no-cache-dir \
    numpy \
    opencv-python \
    matplotlib \
    scikit-image \
    scikit-learn \
    sentence-transformers

# Copier les fichiers de l'application
COPY DESKTOP_APP /opt/DESKTOP_APP

# Commande pour lancer l'application
CMD ["python3", "main.py"]
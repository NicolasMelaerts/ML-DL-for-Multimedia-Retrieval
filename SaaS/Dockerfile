# Utiliser une image de base avec déjà les dépendances système nécessaires
FROM coolsa/pyqt-designer:x64

# Installer les dépendances Python nécessaires
RUN pip install --no-cache-dir \
    numpy \
    opencv-python \
    matplotlib \
    scikit-image \
    scikit-learn \
    sentence-transformers \
    gdown


# Définir le dossier de travail
WORKDIR /opt/SaaS

# Copier les fichiers de requirements d'abord pour profiter du cache Docker
COPY SaaS/requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application
COPY SaaS /opt/SaaS

# Exposer le port Flask
EXPOSE 8080

# Définir les variables d'environnement
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Créer un lien symbolique pour descriptors.py et lancer l'application
CMD ln -sf /opt/DESKTOP_APP/descriptors.py /opt/SaaS/descriptors.py && python3 app.py

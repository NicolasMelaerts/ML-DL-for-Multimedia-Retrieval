# Point de départ : image avec PyQt et X11
FROM coolsa/pyqt-designer:x64

# Installer gdown (pour download depuis Google Drive, si besoin)
RUN pip install --no-cache-dir gdown

# Définir le dossier de travail
WORKDIR /opt/TP

# Copier le script de téléchargement
COPY download_and_unzip.sh .

# Installer les dépendances Python
RUN pip install --no-cache-dir sentence-transformers flask


WORKDIR /opt/TP

COPY . .

# Définir le fichier principal à lancer
CMD ["python3", "main.py"]


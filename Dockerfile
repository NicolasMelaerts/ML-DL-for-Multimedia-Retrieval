# Point de départ : image avec PyQt et X11
FROM coolsa/pyqt-designer:x64

# Installer gdown (pour download depuis Google Drive, si besoin)
RUN pip install --no-cache-dir gdown

COPY download_and_unzip.sh .
RUN chmod +x download_and_unzip.sh && ./download_and_unzip.sh

# Définir le dossier de travail
WORKDIR /opt/TP

# Copier le script de téléchargement
COPY download_and_unzip.sh .

# Installer les dépendances Python
RUN pip install --no-cache-dir sentence-transformers flask


WORKDIR /opt/TP

COPY app.py /opt/TP/app.py
#COPY . .

# Exposer le port Flask
EXPOSE 5000

# Lancer app.py (application Flask)
CMD ["python3", "app.py"]

# Définir le fichier principal à lancer
#CMD ["python3", "main.py"]


# Utilise Python 3.8
FROM python:3.10-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    x11-apps \
    libx11-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-xinerama0 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Installer gdown (pour download depuis Google Drive, si besoin)
RUN pip install --no-cache-dir gdown

# Définir le dossier de travail
WORKDIR /opt/TP

# Copier le script de téléchargement
COPY download_and_unzip.sh .

# Copier le fichier requirements.txt
COPY requirements.txt .

# Exécuter le script de téléchargement (facultatif, voir discussion plus haut)
# RUN chmod +x download_and_unzip.sh && ./download_and_unzip.sh

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Définir le fichier principal à lancer
CMD ["python", "main.py"]

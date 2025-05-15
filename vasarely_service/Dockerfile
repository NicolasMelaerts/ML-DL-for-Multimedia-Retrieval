FROM python:3.9-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    ghostscript \
    python3-tk \
    libx11-6 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail
WORKDIR /app

# Copier les fichiers de requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application
COPY . .

# Exposer le port
EXPOSE 5002

# Commande pour lancer l'application avec Xvfb (serveur X virtuel)
CMD xvfb-run -a python main.py

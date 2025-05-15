# 🎯 ML-DL for Multimedia Retrieval

Ce projet utilise Docker pour encapsuler une application PyQt et un service web Flask. Voici les étapes à suivre pour lancer le projet pour la première fois et les commandes utiles pour les exécutions ultérieures.

---

## 🧰 Prérequis

- Docker installé : https://docs.docker.com/get-docker/
- X11 installé et fonctionnel (pour afficher une interface graphique depuis Docker)
- Sur macOS : XQuartz doit être installé : https://www.xquartz.org/

---

## 🚀 Premier lancement de l'application desktop

### 1. Télécharger les fichiers nécessaires

Exécutez le script `download_and_unzip.sh` pour télécharger la base de données d'images, le dossier de transformer déjà entrainé pour le moteur de recherche par texte, et les features déjà extraites avec Google Colab pour les modèles Deep Learning. Ce script extraira ces fichiers dans le dossier `DESKTOP_APP`.
   
```bash
./download_and_unzip.sh
```

### 2. Configurer l'affichage graphique

#### Sur macOS
```bash
export DISPLAY=192.168.1.40:0.0
```
> Remplacer `192.168.1.40` par l'adresse IP de votre machine (trouvable via `ifconfig`, dans la section `en0`).

- Lancer l'application **XQuartz**
- Aller dans les préférences : `XQuartz > Preferences`
  - Onglet **Security** :
    - Cocher `Allow connections from network clients`
- Redémarrer XQuartz si nécessaire
- Dans le terminal, exécuter :
```bash
xhost +
```

#### Sur Linux
```bash
xhost +local:docker
```

### 3. Construire l'image Docker

```bash
docker build -t desktop_app_image -f DESKTOP_APP/Dockerfile .
```

### 4. Lancer le conteneur et exécuter le programme

```bash
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)/DESKTOP_APP":/opt/DESKTOP_APP \
  desktop_app_image
```

---

## 🔁 Lancement ultérieur de l'application desktop

Si l'image Docker est déjà construite, il suffit de refaire :

#### Sur macOS
```bash
export DISPLAY=192.168.1.40:0.0
xhost +
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)/DESKTOP_APP":/opt/DESKTOP_APP \
  desktop_app_image
```

#### Sur Linux
```bash
xhost +local:docker
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)/DESKTOP_APP":/opt/DESKTOP_APP \
  desktop_app_image
```

---

## 🌐 Lancement du service web (SaaS)

### 1. Construire l'image Docker

```bash
docker build -t flaskapp_v1 -f SaaS/Dockerfile .
```

### 2. Exécuter le conteneur

#### En local
```bash
docker run -d \                                 
  --name flask_app \
  -v "$(pwd)":/opt/TP \
  -w /opt/TP \
  -p 8080:8080 \
  flaskapp_v1 \
  python3 SaaS/app.py
```

### 3. Accéder à l'application

```
http://127.0.0.1:8080 # En local
```

### 4. Redémarrer le conteneur (si nécessaire)

```bash
docker restart flaskapp
```

### 5. Voir les logs

```bash
docker logs flaskapp
```

---

## 🧹 Nettoyage (optionnel)

Pour supprimer les conteneurs :

```bash
docker rm flaskapp
```

Pour supprimer les images Docker :

```bash
docker image rm desktop_app_image flaskapp_v1
```

Pour nettoyer les ressources inutilisées :

```bash
docker system prune
```

---

## 🚀 Lancement du service web (SaaS) avec docker compose

```bash
./run_compose.sh
```


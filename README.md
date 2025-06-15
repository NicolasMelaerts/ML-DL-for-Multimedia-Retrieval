# 🎯 ML-DL for Multimedia Retrieval

---

## 🧰 Prérequis

- Docker installé : https://docs.docker.com/get-docker/
- X11 installé et fonctionnel (pour afficher une interface graphique depuis Docker)
- Sur macOS : XQuartz doit être installé : https://www.xquartz.org/
- Cloner le dépot Git :
```bash
git clone https://github.com/NicolasMelaerts/ML-DL-for-Multimedia-Retrieval.git
cd ML-DL-for-Multimedia-Retrieval
```

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

## 🔁 Lancement ultérieur de l'application desktop en partage de connexion (Iphone host 4G et Mac)

```bash
export DISPLAY=host.docker.internal:0
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)/DESKTOP_APP":/opt/DESKTOP_APP \
  desktop_app_image
```

---

## 🌐 Lancement du service web (SaaS)

### En local : 

```bash
./deploy_local.sh
```

### Adress web : 

```
http://localhost
```


### Sur le serveur :

```bash
./deploy_server.sh
```

### Adress web : 

```
http://163.172.234.110
```

### Pour nettoyer les ressources inutilisées :

```bash
docker system prune
```

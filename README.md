# 🎯 ML-DL for Multimedia Retrieval

Ce projet utilise Docker pour encapsuler une application PyQt. Voici les étapes à suivre pour lancer le projet pour la première fois et les commandes utiles pour les exécutions ultérieures.

---

## 🧰 Prérequis

- Docker installé : https://docs.docker.com/get-docker/
- X11 installé et fonctionnel (pour afficher une interface graphique depuis Docker)
- Sur macOS : XQuartz doit être installé : https://www.xquartz.org/

---

## 🚀 Premier lancement

### 1. Démarrer XQuartz

- Lancer l’application **XQuartz**
- Aller dans les préférences : `XQuartz > Preferences`
  - Onglet **Security** :
    - Cocher `Allow connections from network clients`
- Redémarrer XQuartz si nécessaire
- Dans le terminal, exécuter :

```bash
xhost +
```

---

### 2. Définir la variable d’environnement DISPLAY

```bash
export DISPLAY=192.168.1.40:0.0
```

> Remplacer `192.168.1.40` par l’adresse IP de votre machine (trouvable via `ifconfig`, dans la section `en0`).

---

### 3. Construire l’image Docker

Depuis le répertoire du projet (où se trouve le Dockerfile) :

```bash
docker build -t my_project .
```

---

### 4. Lancer le conteneur et exécuter le programme

```bash
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)":/opt/TP -w /opt/TP my_project bash
```

---

## 🔁 Lancement ultérieur

Si l’image Docker est déjà construite, il suffit de refaire :

```bash
export DISPLAY=192.168.1.40:0.0
xhost +
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)":/opt/TP -w /opt/TP my_project bash
```

Puis dans le conteneur :

1. **Télécharger les fichiers nécessaires :**

   À l'intérieur du conteneur, exécutez le script `download_and_unzip.sh` pour télécharger la base de données d'images, le dossier de transformer déjà entrainé pour le moteur de recherche par texte, et les features déjà extraites avec google colabpour les modèles Deep Learning. Ce script extraira ces fichiers dans le dossier `DESKTOP_APP`.
   
   ```bash
   ./download_and_unzip.sh
   ```

2. **Entrez dans le dossier `DESKTOP_APP` :**

```bash
cd DESKTOP_APP
```

3. **Lancez l'application :**

```bash
python3 main.py
```

---

## 🧹 Nettoyage (optionnel)

Pour supprimer l’image Docker :

```bash
docker image rm my_project
```

Pour nettoyer les ressources inutilisées :

```bash
docker system prune
```

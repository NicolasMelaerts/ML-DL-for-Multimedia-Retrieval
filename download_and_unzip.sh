#!/bin/bash

# ID du fichier Google Drive
FILE_ID="1APY6Ze-PG60hDD81EhuW71s_WtqA_qkg"
FILENAME="Projet_moteur_recherche.zip"
DEST_DIR="DESKTOP_APP"

# Télécharger avec gdown
gdown "$FILE_ID" -O "$FILENAME"

# Décompresser
if [ -f "$FILENAME" ]; then
    echo "Téléchargement terminé : $FILENAME"
    
    # Créer un dossier temporaire pour l'extraction
    TEMP_DIR="temp_extraction"
    mkdir -p "$TEMP_DIR"
    
    # Extraire dans le dossier temporaire
    unzip -o "$FILENAME" -d "$TEMP_DIR"
    
    # Déplacer le contenu dans le dossier DEST_DIR
    INNER_DIR="$TEMP_DIR/Projet_moteur_recherche"
    if [ -d "$INNER_DIR" ]; then
        mkdir -p "$DEST_DIR"
        mv "$INNER_DIR"/* "$DEST_DIR"
        echo "Contenu déplacé dans le dossier $DEST_DIR/"
    else
        echo "Erreur : le dossier Projet_moteur_recherche n'a pas été trouvé dans l'archive."
    fi
    
    # Supprimer le dossier temporaire et le zip
    rm -rf "$TEMP_DIR" "$FILENAME"
    
    echo "Décompression terminée."
else
    echo "Échec du téléchargement."
fi

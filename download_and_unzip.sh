#!/bin/bash

# ID du fichier Google Drive
FILE_ID="1APY6Ze-PG60hDD81EhuW71s_WtqA_qkg"
FILENAME="Projet_moteur_recherche.zip"

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
    
    # Déplacer le contenu de Projet_moteur_recherche/ vers le dossier courant
    INNER_DIR="$TEMP_DIR/Projet_moteur_recherche"
    if [ -d "$INNER_DIR" ]; then
        mv "$INNER_DIR"/* .
    else
        echo "Erreur : le dossier Projet_moteur_recherche n'a pas été trouvé dans l'archive."
    fi
    
    # Supprimer le dossier temporaire et le zip
    rm -rf "$TEMP_DIR" "$FILENAME"
    
    echo "Décompression terminée. Contenu extrait dans le dossier courant."
else
    echo "Échec du téléchargement."
fi

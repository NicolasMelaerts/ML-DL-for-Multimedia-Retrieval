#!/bin/bash
docker compose down      # Arrête et supprime les conteneurs
docker compose rm -f     # Force la suppression des conteneurs arrêtés
docker compose build     # Recrée les images Docker
docker compose up        # Démarre les conteneurs en mode interactif

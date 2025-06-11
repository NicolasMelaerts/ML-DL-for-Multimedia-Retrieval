# -*- coding: utf-8 -*-
"""
Routes pour la recherche d'images par texte
"""

from flask import Blueprint, render_template, request
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import base64
from io import BytesIO
import time

text_search_bp = Blueprint("text_search", __name__)

# Chemin vers le dossier DESKTOP_APP
DESKTOP_APP_PATH = "/opt/DESKTOP_APP"

# Configuration
MODEL_PATH = os.path.join(DESKTOP_APP_PATH, "Transformer/sentence_transformer_model")
CAPTIONS_FILE = os.path.join(DESKTOP_APP_PATH, "Transformer/captions.json")
EMBEDDINGS_DIR = os.path.join(DESKTOP_APP_PATH, "Transformer/embeddings_output")
DATASETS_DIR = os.path.join(DESKTOP_APP_PATH, "MIR_DATASETS_B")

# Configuration des animaux et races pour la recherche rapide d'images
animaux = ["araignee", "chiens", "oiseaux", "poissons", "singes"]
araignees = ["barn spider", "garden spider", "orb-weaving spider", "tarantula", "trap_door spider", "wolf spider"]
chiens = ["boxer", "Chihuahua", "golden\x20retriever", "Labrador\x20retriever", "Rottweiler", "Siberian\x20husky"]
oiseaux = ["blue jay", "bulbul", "great grey owl", "parrot", "robin", "vulture"]
poissons = ["dogfish", "eagle ray", "guitarfish", "hammerhead", "ray", "tiger shark"]
singes = ["baboon", "chimpanzee", "gorilla", "macaque", "orangutan", "squirrel monkey"]

# Variables globales
model = None
captions = {}

@text_search_bp.route('/', methods=['GET', 'POST'])
def index():
    global model, captions
    
    # Initialiser les variables
    results = []
    error_message = ""
    success_message = ""
    query = ""
    top_k = 5
    performance_info = {}
    
    # Charger le modèle si ce n'est pas déjà fait
    if model is None:
        try:
            model = SentenceTransformer(MODEL_PATH)
            success_message = "Modèle chargé avec succès."
        except Exception as e:
            error_message = f"Erreur lors du chargement du modèle: {str(e)}"
    
    # Charger les descriptions si ce n'est pas déjà fait
    if not captions and os.path.exists(CAPTIONS_FILE):
        try:
            with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
                captions = json.load(f)
            success_message = "Descriptions chargées avec succès."
        except Exception as e:
            error_message = f"Erreur lors du chargement des descriptions: {str(e)}"
    
    # Traiter la recherche
    if request.method == 'POST':
        query = request.form.get('query', '')
        top_k = int(request.form.get('top_k', 5))
        
        if not query:
            error_message = "Veuillez entrer une description."
        elif model is None:
            error_message = "Le modèle n'est pas chargé."
        else:
            try:
                # Mesurer le temps total
                total_start_time = time.time()
                
                # Encoder la requête
                encoding_start = time.time()
                query_embedding = model.encode(query)
                encoding_time = time.time() - encoding_start
                performance_info["encoding_time"] = round(encoding_time, 4)
                
                # Recherche des images les plus proches
                embedding_search_start = time.time()
                similarity_calc_total = 0
                image_path_search_total = 0
                file_count = 0
                results_list = []
                
                for root, dirs, files in os.walk(EMBEDDINGS_DIR):
                    for file in files:
                        if file.endswith('_embedding.txt'):
                            file_count += 1
                            emb_path = os.path.join(root, file)
                            try:
                                # Charger l'embedding
                                emb = np.fromstring(open(emb_path).read(), sep=' ')
                                
                                # Calculer la similarité
                                sim_start = time.time()
                                sim = cosine_similarity([query_embedding], [emb])[0][0]
                                similarity_calc_total += time.time() - sim_start
                                
                                # Extraire le chemin relatif
                                relative_path = os.path.relpath(emb_path, EMBEDDINGS_DIR)
                                relative_path = relative_path.replace('_embedding.txt', '')
                                
                                # Extraire l'animal et la race
                                path_parts = relative_path.split('/')
                                animal = path_parts[0] if len(path_parts) > 0 else None
                                race = path_parts[1] if len(path_parts) > 1 else None
                                
                                # Construire le chemin de l'image
                                image_filename = os.path.basename(relative_path)
                                
                                # Utiliser notre fonction optimisée pour trouver le chemin de l'image
                                path_search_start = time.time()
                                image_path = find_image_path(image_filename, animal, race)
                                image_path_search_total += time.time() - path_search_start
                                
                                # Récupérer la description si disponible
                                caption = ""
                                for key in captions:
                                    if image_filename in key:
                                        caption = captions[key]
                                        break
                                
                                # Ajouter aux résultats seulement si l'image est trouvée
                                if image_path:
                                    results_list.append((image_path, caption, sim, animal, race))
                            except Exception as e:
                                print(f"Erreur lors du traitement de {emb_path}: {str(e)}")
                
                embedding_search_time = time.time() - embedding_search_start
                performance_info["embedding_search_time"] = round(embedding_search_time, 4)
                performance_info["similarity_calc_total"] = round(similarity_calc_total, 4)
                performance_info["image_path_search_total"] = round(image_path_search_total, 4)
                performance_info["file_count"] = file_count
                
                # Trier les résultats par similarité décroissante
                sort_start = time.time()
                results_list.sort(key=lambda x: x[2], reverse=True)
                sort_time = time.time() - sort_start
                performance_info["sort_time"] = round(sort_time, 4)
                
                # Limiter aux top_k résultats
                results_list = results_list[:top_k]
                
                # Préparation des résultats pour l'affichage
                display_start = time.time()
                for image_path, caption, sim, animal, race in results_list:
                    # Convertir l'image en base64 pour l'affichage
                    image_data = None
                    if image_path and os.path.exists(image_path):
                        with Image.open(image_path) as img:
                            img = img.resize((200, 200), Image.LANCZOS)
                            buffer = BytesIO()
                            img.save(buffer, format="JPEG")
                            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    # Ajouter aux résultats
                    results.append({
                        'image_path': image_path,
                        'image_data': image_data,
                        'caption': caption,
                        'similarity': sim,
                        'animal': animal,
                        'race': race
                    })
                
                display_time = time.time() - display_start
                performance_info["display_time"] = round(display_time, 4)
                
                # Temps total
                total_time = time.time() - total_start_time
                performance_info["total_time"] = round(total_time, 4)
                
                success_message = f"Recherche effectuée avec succès en {performance_info['total_time']} secondes."
            except Exception as e:
                error_message = f"Erreur lors de la recherche: {str(e)}"
    
    return render_template('text_search.html', 
                          title="Recherche d'Images par Texte",
                          results=results, 
                          error_message=error_message, 
                          success_message=success_message,
                          query=query,
                          top_k=top_k,
                          performance_info=performance_info)

# Fonction pour trouver la race correspondante dans la liste
def find_matching_race(race_name, races_list):
    """
    Trouve la race correspondante dans la liste, en tenant compte des différents formats
    """
    if not races_list:
        return None
    
    # Normaliser le nom de race du fichier
    race_name_lower = race_name.lower()
    
    # 1. Essayer une correspondance directe
    for race in races_list:
        if race.lower() == race_name_lower:
            return race
    
    # 2. Essayer en remplaçant les espaces par des underscores
    for race in races_list:
        if race.lower().replace(' ', '_') == race_name_lower:
            return race
    
    # 3. Essayer en supprimant les espaces
    for race in races_list:
        if race.lower().replace(' ', '') == race_name_lower:
            return race
    
    # 4. Essayer une correspondance partielle
    for race in races_list:
        race_lower = race.lower()
        if race_name_lower in race_lower or race_lower in race_name_lower:
            return race
        
        # Essayer aussi sans les espaces
        race_lower_no_spaces = race_lower.replace(' ', '')
        if race_name_lower in race_lower_no_spaces or race_lower_no_spaces in race_name_lower:
            return race
    
    # Si aucune correspondance n'est trouvée, retourner la première race de la liste
    return races_list[0]

# Version optimisée de la recherche d'images (basée sur text_search_page.py)
def find_image_path(image_filename, animal=None, race=None):
    """
    Trouve le chemin d'une image en utilisant directement les informations d'animal et de race
    """
    # Si animal et race sont fournis, utiliser directement ces informations
    if animal and race:
        # Essayer différentes extensions
        for img_ext in ['.jpg', '.jpeg', '.png']:
            # Extraire le nom de base sans extension
            base_name = os.path.splitext(image_filename)[0]
            direct_path = os.path.join(DATASETS_DIR, animal, race, f"{base_name}{img_ext}")
            if os.path.exists(direct_path):
                return direct_path
        
        # Si aucune correspondance exacte, essayer de trouver la race correspondante
        races_list = None
        if animal == "araignee":
            races_list = araignees
        elif animal == "chiens":
            races_list = chiens
        elif animal == "oiseaux":
            races_list = oiseaux
        elif animal == "poissons":
            races_list = poissons
        elif animal == "singes":
            races_list = singes
        
        if races_list:
            matching_race = find_matching_race(race, races_list)
            if matching_race:
                for img_ext in ['.jpg', '.jpeg', '.png']:
                    base_name = os.path.splitext(image_filename)[0]
                    direct_path = os.path.join(DATASETS_DIR, animal, matching_race, f"{base_name}{img_ext}")
                    if os.path.exists(direct_path):
                        return direct_path
    
    # Si on arrive ici ou si animal/race ne sont pas fournis, parcourir tous les animaux et races
    # Extraire le nom de base sans extension
    base_name = os.path.splitext(image_filename)[0]
    
    # Parcourir tous les animaux et races
    for animal in animaux:
        races_list = None
        if animal == "araignee":
            races_list = araignees
        elif animal == "chiens":
            races_list = chiens
        elif animal == "oiseaux":
            races_list = oiseaux
        elif animal == "poissons":
            races_list = poissons
        elif animal == "singes":
            races_list = singes
        
        if races_list:
            for race in races_list:
                # Essayer de trouver l'image dans ce dossier
                for img_ext in ['.jpg', '.jpeg', '.png']:
                    image_name = f"{base_name}{img_ext}"
                    direct_path = os.path.join(DATASETS_DIR, animal, race, image_name)
                    if os.path.exists(direct_path):
                        return direct_path
    
    # Si on arrive ici, on n'a pas trouvé le fichier
    return None 
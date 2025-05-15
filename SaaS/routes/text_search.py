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

text_search_bp = Blueprint("text_search", __name__)

# Chemin vers le dossier DESKTOP_APP
DESKTOP_APP_PATH = "/opt/DESKTOP_APP"

# Configuration
MODEL_PATH = os.path.join(DESKTOP_APP_PATH, "Transformer/sentence_transformer_model")
CAPTIONS_FILE = os.path.join(DESKTOP_APP_PATH, "Transformer/captions.json")
EMBEDDINGS_DIR = os.path.join(DESKTOP_APP_PATH, "Transformer/embeddings_output")
DATASETS_DIR = os.path.join(DESKTOP_APP_PATH, "MIR_DATASETS_B")

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
    top_k = 2
    
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
                # Encoder la requête
                query_embedding = model.encode(query)
                
                # Recherche des images les plus proches
                for root, dirs, files in os.walk(EMBEDDINGS_DIR):
                    for file in files:
                        if file.endswith('_embedding.txt'):
                            emb_path = os.path.join(root, file)
                            try:
                                # Charger l'embedding
                                emb = np.fromstring(open(emb_path).read(), sep=' ')
                                
                                # Calculer la similarité
                                sim = cosine_similarity([query_embedding], [emb])[0][0]
                                
                                # Extraire le chemin relatif
                                relative_path = os.path.relpath(emb_path, EMBEDDINGS_DIR)
                                relative_path = relative_path.replace('_embedding.txt', '')
                                
                                # Extraire l'animal et la race
                                path_parts = relative_path.split('/')
                                animal = path_parts[0] if len(path_parts) > 0 else "Inconnu"
                                race = path_parts[1] if len(path_parts) > 1 else "Inconnue"
                                
                                # Construire le chemin de l'image
                                image_filename = os.path.basename(relative_path)
                                
                                # Utiliser notre fonction optimisée pour trouver le chemin de l'image
                                image_path = find_image_path(image_filename)
                                
                                # Récupérer la description si disponible
                                caption = ""
                                for key in captions:
                                    if image_filename in key:
                                        caption = captions[key]
                                        break
                                
                                # Convertir l'image en base64 pour l'affichage
                                image_data = None
                                if image_path and os.path.exists(image_path):
                                    with Image.open(image_path) as img:
                                        img = img.resize((200, 200), Image.LANCZOS)
                                        buffer = BytesIO()
                                        img.save(buffer, format="JPEG")
                                        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                
                                # Ajouter aux résultats
                                attempted_path = image_path if image_path else f"Image non trouvée. Chemin tenté: {os.path.join(DATASETS_DIR, animal, race, image_filename)}"
                                results.append({
                                    'image_path': attempted_path,
                                    'image_data': image_data,
                                    'caption': caption,
                                    'similarity': sim,
                                    'animal': animal,
                                    'race': race
                                })
                            except Exception as e:
                                print(f"Erreur lors du traitement de {emb_path}: {str(e)}")
                
                # Trier les résultats par similarité décroissante
                results.sort(key=lambda x: x['similarity'], reverse=True)
                
                # Limiter aux top_k résultats
                results = results[:top_k]
                
                success_message = "Recherche effectuée avec succès."
            except Exception as e:
                error_message = f"Erreur lors de la recherche: {str(e)}"
    
    return render_template('text_search.html', 
                          title="Recherche d'Images par Texte",
                          results=results, 
                          error_message=error_message, 
                          success_message=success_message,
                          query=query,
                          top_k=top_k) 

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

# Optimisation de la recherche d'images en supposant que toutes les images existent
def find_image_path(image_filename):
    """
    Trouve le chemin d'une image en utilisant la structure connue du nom de fichier
    Format: X_Y_animal_race_ZZZZ.jpg
    Suppose que toutes les images existent dans la base de données
    """
    # Vérifier si le nom de fichier contient déjà l'extension
    if not image_filename.lower().endswith('.jpg'):
        # Ajouter l'extension .jpg si elle n'est pas présente
        image_filename = image_filename + '.jpg'
    
    # Extraire les parties du nom de fichier (sans l'extension)
    base_name = os.path.splitext(image_filename)[0]
    parts = base_name.split('_')
    
    if len(parts) < 5:  # Vérifier qu'il y a assez de parties
        return None
    
    # Les parties 3 et 4 sont l'animal et la race
    animal_name = parts[2]
    race_name = parts[3]
    
    # Vérifier si l'animal est dans notre liste
    if animal_name in animaux:
        # Déterminer la liste de races correspondante
        races_list = None
        if animal_name == "araignee":
            races_list = araignees
        elif animal_name == "chiens":
            races_list = chiens
        elif animal_name == "oiseaux":
            races_list = oiseaux
        elif animal_name == "poissons":
            races_list = poissons
        elif animal_name == "singes":
            races_list = singes
        
        # Trouver la race correspondante
        matching_race = find_matching_race(race_name, races_list)
        if matching_race:
            # Construire le chemin avec le nom exact de la race (avec espaces)
            direct_path = os.path.join(DATASETS_DIR, animal_name, matching_race, image_filename)
            if os.path.exists(direct_path):
                return direct_path
    
    # Si on arrive ici, c'est qu'on n'a pas trouvé de correspondance exacte
    # Trouver l'animal le plus proche
    closest_animal = None
    for animal in animaux:
        if animal_name.lower() in animal.lower() or animal.lower() in animal_name.lower():
            closest_animal = animal
            break
    
    if not closest_animal:
        closest_animal = animal_name  # Utiliser tel quel si pas de correspondance
    
    # Déterminer la liste de races correspondante
    races_list = None
    if closest_animal == "araignee":
        races_list = araignees
    elif closest_animal == "chiens":
        races_list = chiens
    elif closest_animal == "oiseaux":
        races_list = oiseaux
    elif closest_animal == "poissons":
        races_list = poissons
    elif closest_animal == "singes":
        races_list = singes
    
    # Trouver la race correspondante
    matching_race = find_matching_race(race_name, races_list)
    if matching_race:
        # Construire le chemin avec le nom exact de la race (avec espaces)
        direct_path = os.path.join(DATASETS_DIR, closest_animal, matching_race, image_filename)
        if os.path.exists(direct_path):
            return direct_path
    
    # Si on arrive ici, on n'a pas trouvé le fichier
    return None 
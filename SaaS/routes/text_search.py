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

text_search_bp = Blueprint('text_search', __name__)

# Configuration
MODEL_PATH = "DESKTOP_APP/Transformer/sentence_transformer_model"
CAPTIONS_FILE = "DESKTOP_APP/Transformer/captions.json"
EMBEDDINGS_DIR = "DESKTOP_APP/Transformer/embeddings_output"
DATASETS_DIR = "MIR_DATASETS_B"

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
                                
                                # Rechercher l'image récursivement dans MIR_DATASETS_B
                                image_path = None
                                for img_root, img_dirs, img_files in os.walk(DATASETS_DIR):
                                    for img_file in img_files:
                                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')) and image_filename in img_file:
                                            image_path = os.path.join(img_root, img_file)
                                            break
                                    if image_path:
                                        break
                                
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
                                results.append({
                                    'image_path': image_path if image_path else "Image non trouvée",
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
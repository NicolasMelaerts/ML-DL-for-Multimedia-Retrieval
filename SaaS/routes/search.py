from flask import Blueprint, render_template, request, flash, redirect, url_for
import os
import sys
import cv2
import numpy as np
import base64
import time
import glob
import matplotlib.pyplot as plt
import io
from io import BytesIO
from PIL import Image
from utils.decorators import login_required

# Chemin vers le dossier DESKTOP_APP
DESKTOP_APP_PATH = "/opt/DESKTOP_APP"
sys.path.append(DESKTOP_APP_PATH)

# Dossier contenant la base d'images
IMAGE_FOLDER = os.path.join(DESKTOP_APP_PATH, "MIR_DATASETS_B")

# Dossier pour les fichiers temporaires
TEMP_FOLDER = os.path.join(DESKTOP_APP_PATH, "temp")

# Dossier contenant les descripteurs
DESCRIPTORS_FOLDER = os.path.join(DESKTOP_APP_PATH, "Descripteurs")

# Importer les fonctions nécessaires
from descriptors import extractReqFeatures
from distances import getkVoisins

search_bp = Blueprint('search', __name__)

# Configuration des animaux et races pour la recherche rapide d'images
animaux = ["araignee", "chiens", "oiseaux", "poissons", "singes"]
araignees = ["barn spider", "garden spider", "orb-weaving spider", "tarantula", "trap_door spider", "wolf spider"]
chiens = ["boxer", "Chihuahua", "golden\x20retriever", "Labrador\x20retriever", "Rottweiler", "Siberian\x20husky"]
oiseaux = ["blue jay", "bulbul", "great grey owl", "parrot", "robin", "vulture"]
poissons = ["dogfish", "eagle ray", "guitarfish", "hammerhead", "ray", "tiger shark"]
singes = ["baboon", "chimpanzee", "gorilla", "macaque", "orangutan", "squirrel monkey"]

@search_bp.route('/', methods=['GET', 'POST'])
@login_required
def index():
    results = []
    query_image_data = None
    performance_info = {}
    metrics = None
    descriptor_status = check_available_descriptors()
    
    # Au début de la fonction index(), ajouter ces variables pour le template
    races = {
        'araignee': araignees,
        'chiens': chiens,
        'oiseaux': oiseaux,
        'poissons': poissons,
        'singes': singes
    }
    
    if request.method == 'POST':
        # Mesurer le temps total
        total_start_time = time.time()
        
        # Récupérer les descripteurs sélectionnés
        selected_descriptors = []
        if request.form.get('bgr'): selected_descriptors.append('BGR')
        if request.form.get('hsv'): selected_descriptors.append('HSV')
        if request.form.get('glcm'): selected_descriptors.append('GLCM')
        if request.form.get('hog'): selected_descriptors.append('HOG')
        if request.form.get('lbp'): selected_descriptors.append('LBP')
        if request.form.get('orb'): selected_descriptors.append('ORB')
        
        if not selected_descriptors:
            flash("Veuillez sélectionner au moins un descripteur.", "warning")
            return redirect(url_for('search.index'))
        
        # Récupérer la distance sélectionnée
        distance = request.form.get('distance', 'Euclidienne')
        
        # Récupérer le nombre de résultats à afficher
        top_k = int(request.form.get('top_k', 20))
        
        # Récupérer la classe attendue à partir du chemin de l'image requête
        expected_class = ''
        
        # Traiter l'image téléchargée
        if 'query_image' in request.files:
            query_file = request.files['query_image']
            if query_file.filename != '':
                # Mesurer le temps de traitement de l'image
                image_processing_start = time.time()
                
                # Sauvegarder temporairement l'image
                temp_path = os.path.join(TEMP_FOLDER, 'temp_query.jpg')
                
                # Créer le dossier temporaire s'il n'existe pas
                if not os.path.exists(TEMP_FOLDER):
                    os.makedirs(TEMP_FOLDER)
                
                query_file.save(temp_path)
                
                # Essayer d'extraire la classe à partir du nom du fichier
                filename = query_file.filename
                parts = filename.split('/')
                if len(parts) >= 3:
                    # Format attendu: animal/race/image.jpg
                    expected_class = f"{parts[-3]}/{parts[-2]}"
                elif len(parts) == 2:
                    expected_class = f"{parts[0]}/{parts[1]}"
                
                # Si le chemin ne contient pas la structure attendue, essayer de déduire à partir du nom
                if not expected_class:
                    # Analyser le nom du fichier pour voir s'il contient des indices sur la classe
                    filename_without_ext = os.path.splitext(filename)[0]
                    for animal in animaux:
                        for race_list_name in ['araignees', 'chiens', 'oiseaux', 'poissons', 'singes']:
                            race_list = globals()[race_list_name]
                            if animal in filename_without_ext.lower():
                                for race in race_list:
                                    if race.lower() in filename_without_ext.lower():
                                        expected_class = f"{animal}/{race}"
                                        break
                                if expected_class:
                                    break
                        if expected_class:
                            break
                
                # Charger l'image pour l'affichage
                with open(temp_path, 'rb') as img_file:
                    query_image_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                image_processing_time = time.time() - image_processing_start
                performance_info["image_processing_time"] = round(image_processing_time, 4)
                
                # Charger les descripteurs et effectuer la recherche
                features_dict = {}
                all_results = {}
                min_scores = {}
                max_scores = {}
                
                # Mesurer le temps de chargement des descripteurs
                features_loading_start = time.time()
                
                for desc_type in selected_descriptors:
                    try:
                        # Charger les descripteurs
                        features = load_features(desc_type)
                        if features:
                            features_dict[desc_type] = features
                            
                            # Extraire les caractéristiques de l'image requête
                            algo_choice = get_algo_choice(desc_type)
                            req_features = extractReqFeatures(temp_path, algo_choice)
                            
                            # Vérifier la compatibilité des dimensions
                            if features and len(features) > 0:
                                sample_feature = features[0][1]
                                if hasattr(req_features, 'shape') and hasattr(sample_feature, 'shape'):
                                    if req_features.shape != sample_feature.shape:
                                        req_features = np.resize(req_features, sample_feature.shape)
                            
                            # Adapter la distance pour ce descripteur
                            current_distance = adapt_distance_for_descriptor(desc_type, distance)
                            
                            # Rechercher les voisins
                            neighbors = getkVoisins(features, req_features, top_k, current_distance)
                            
                            # Initialiser les min/max pour ce descripteur
                            if neighbors:
                                min_score = float('inf')
                                max_score = float('-inf')
                                
                                # Ajouter les résultats au dictionnaire global
                                for path, _, dist in neighbors:
                                    if path not in all_results:
                                        all_results[path] = {}
                                    
                                    # Stocker le score pour ce descripteur
                                    score = dist
                                    all_results[path][desc_type] = score
                                    
                                    # Mettre à jour min/max
                                    min_score = min(min_score, score)
                                    max_score = max(max_score, score)
                                
                                # Stocker min/max pour ce descripteur
                                min_scores[desc_type] = min_score
                                max_scores[desc_type] = max_score
                    except Exception as e:
                        flash(f"Erreur lors de la recherche avec {desc_type}: {str(e)}", "error")
                
                features_loading_time = time.time() - features_loading_start
                performance_info["features_loading_time"] = round(features_loading_time, 4)
                
                # Combiner les scores pour tous les descripteurs
                search_start = time.time()
                combined_results = []
                
                for path, scores in all_results.items():
                    # Calculer un score combiné normalisé
                    combined_score = 0
                    num_descriptors = 0
                    contributed_descriptors = []
                    
                    for desc_type, score in scores.items():
                        # Normaliser le score entre 0 et 1 (0 = meilleur, 1 = pire)
                        min_score = min_scores.get(desc_type, 0)
                        max_score = max_scores.get(desc_type, 1)
                        score_range = max_score - min_score
                        
                        if score_range > 0:
                            # Pour les mesures de distance (plus petit = meilleur)
                            if distance not in ["Cosinus", "Correlation", "Intersection"]:
                                normalized_score = (score - min_score) / score_range
                            else:
                                # Pour les mesures de similarité (plus grand = meilleur)
                                normalized_score = 1 - (score - min_score) / score_range
                            
                            # Ajouter le score normalisé
                            combined_score += normalized_score
                            num_descriptors += 1
                            contributed_descriptors.append(desc_type)
                    
                    # Calculer la moyenne des scores normalisés
                    if num_descriptors > 0:
                        avg_score = combined_score / num_descriptors
                        combined_results.append((path, avg_score, "+".join(contributed_descriptors)))
                
                # Trier les résultats par score combiné
                combined_results.sort(key=lambda x: x[1])
                
                # Prendre les top_k premiers résultats
                results_list = combined_results[:top_k]
                
                search_time = time.time() - search_start
                performance_info["search_time"] = round(search_time, 4)
                
                # Préparer les résultats pour l'affichage
                display_start = time.time()
                
                formatted_results = []
                for i, (path, dist, desc_type) in enumerate(results_list):
                    try:
                        # Charger l'image en base64
                        with open(path, 'rb') as img_file:
                            img = Image.open(img_file)
                            img = img.resize((200, 200), Image.LANCZOS)
                            buffer = BytesIO()
                            img.save(buffer, format="JPEG")
                            img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        # Extraire la classe (animal/breed)
                        parts = path.split(os.sep)
                        if len(parts) >= 3:
                            # Format attendu: .../animal/race/image.jpg
                            animal_idx = max(0, len(parts) - 3)
                            breed_idx = max(0, len(parts) - 2)
                            img_class = f"{parts[animal_idx]}/{parts[breed_idx]}"
                        else:
                            img_class = "unknown"
                        
                        # Vérifier si l'image est pertinente (de la classe attendue)
                        is_relevant = False
                        if expected_class and expected_class.strip():
                            is_relevant = (img_class.lower() == expected_class.lower())
                        
                        formatted_results.append({
                            'rank': i + 1,
                            'name': os.path.basename(path),
                            'class': img_class,
                            'distance': dist,
                            'descriptors': desc_type,
                            'image_data': img_data,
                            'relevant': is_relevant
                        })
                    except Exception as e:
                        flash(f"Erreur lors du traitement de l'image {path}: {str(e)}", "error")
                
                display_time = time.time() - display_start
                performance_info["display_time"] = round(display_time, 4)
                
                # Supprimer l'image temporaire
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                results = formatted_results
                
                # Calculer les métriques d'évaluation si une classe attendue est fournie
                if expected_class and expected_class.strip() and results:
                    metrics = calculate_metrics(results, expected_class)
                
                # Temps total
                total_time = time.time() - total_start_time
                performance_info["total_time"] = round(total_time, 4)
    
    return render_template(
        'search.html',
        title="Moteur de Recherche avec descripteurs",
        results=results,
        query_image=query_image_data,
        performance_info=performance_info,
        descriptor_status=descriptor_status,
        metrics=metrics,
        animaux=animaux,
        races=races,
        expected_class=expected_class
    )

def check_available_descriptors():
    """Vérifie quels descripteurs sont disponibles"""
    descriptor_status = {
        'BGR': False,
        'HSV': False,
        'GLCM': False,
        'HOG': False,
        'LBP': False,
        'ORB': False
    }
    
    if os.path.exists(DESCRIPTORS_FOLDER):
        for desc_type in descriptor_status.keys():
            desc_folder = os.path.join(DESCRIPTORS_FOLDER, desc_type)
            if os.path.isdir(desc_folder) and any(f.endswith('.txt') for f in os.listdir(desc_folder)):
                descriptor_status[desc_type] = True
    
    return descriptor_status

def get_algo_choice(desc_type):
    """Retourne l'algo_choice correspondant au type de descripteur"""
    mapping = {
        'BGR': 1,
        'HOG': 2,
        'LBP': 3,
        'ORB': 4,
        'HSV': 5,
        'GLCM': 6
    }
    return mapping.get(desc_type, 0)

def adapt_distance_for_descriptor(desc_type, distance_name):
    """Adapte la mesure de distance au type de descripteur"""
    if desc_type == 'ORB' and distance_name not in ["Brute force", "Flann"]:
        return "Brute force"
    elif desc_type in ['BGR', 'HSV'] and distance_name in ["Chi carre", "Intersection", "Bhattacharyya", "Correlation"]:
        return distance_name
    return distance_name

def find_image_in_directory(base_dir, image_name):
    """
    Recherche récursivement une image dans un dossier et ses sous-dossiers.
    """
    # Essayer différentes extensions
    for img_ext in ['.jpg', '.jpeg', '.png']:
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
                    image_path = os.path.join(base_dir, animal, race, f"{image_name}{img_ext}")
                    if os.path.exists(image_path):
                        return image_path
    
    # Si on arrive ici, faire une recherche plus générale
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_name_without_ext = os.path.splitext(file)[0]
                if file_name_without_ext == image_name:
                    return os.path.join(root, file)
    return None

def load_features(desc_type):
    """Charge les descripteurs d'un type spécifique"""
    features = []
    
    # Construire le chemin complet vers le sous-dossier du descripteur
    folder_path = os.path.join(DESCRIPTORS_FOLDER, desc_type)
    
    # Vérifier si le dossier existe
    if not os.path.exists(folder_path):
        print(f"Le dossier {folder_path} n'existe pas.")
        return []
    
    # Obtenir la liste de tous les fichiers .txt dans le dossier
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    # Utiliser un dictionnaire pour accélérer la recherche d'images
    image_path_cache = {}
    
    # Traiter les fichiers
    algo_id = get_algo_choice(desc_type)
    
    for file_name in all_files:
        try:
            # Charger le descripteur
            data_path = os.path.join(folder_path, file_name)
            feature = np.loadtxt(data_path)
            
            # Extraire les informations du nom de fichier
            parts = os.path.splitext(file_name)[0].split('_')
            
            # Format 1: Methode_1_animal_race_imagename.txt
            if len(parts) >= 4 and parts[0] == "Methode":
                animal = parts[2]
                breed = parts[3]
                image_name = '_'.join(parts[4:]) + '.jpg'
            # Format 2: animal_race_imagename.txt
            elif len(parts) >= 2:
                animal = parts[0]
                breed = parts[1]
                image_name = '_'.join(parts[2:]) + '.jpg'
            else:
                print(f"Format de nom de fichier non reconnu: {file_name}")
                continue
            
            # Construire le chemin de l'image
            image_path = os.path.join(IMAGE_FOLDER, animal, breed, image_name)
            
            # Vérifier si le chemin existe déjà dans le cache
            if image_name in image_path_cache:
                image_path = image_path_cache[image_name]
                features.append((image_path, feature))
                continue
            
            if os.path.exists(image_path):
                # Ajouter au cache
                image_path_cache[image_name] = image_path
                features.append((image_path, feature))
            else:
                # Essayer de trouver l'image avec la fonction find_image_in_directory
                image_name_without_ext = os.path.splitext(image_name)[0]
                found_path = find_image_in_directory(IMAGE_FOLDER, image_name_without_ext)
                if found_path:
                    # Ajouter au cache
                    image_path_cache[image_name] = found_path
                    features.append((found_path, feature))
        
        except Exception as e:
            print(f"Erreur lors du chargement de {file_name}: {str(e)}")
    
    return features

def calculate_metrics(results, expected_class):
    """
    Calcule les métriques d'évaluation à partir des résultats de recherche.
    """
    metrics = {}
    
    # Compter le nombre total d'images pertinentes dans la base
    # Pour simplifier, nous utilisons une valeur approximative
    # Dans une implémentation réelle, cela devrait être calculé en fonction de la base de données
    total_relevant = 20  # Supposons qu'il y a environ 20 images par classe
    
    # Nombre de résultats
    n = len(results)
    
    # Compter les vrais positifs à chaque rang
    tp_at_rank = [0] * n
    for i, result in enumerate(results):
        if result.get('relevant', False):
            tp_at_rank[i] = 1
    
    # Calculer les TP cumulés
    tp_cumul = [sum(tp_at_rank[:i+1]) for i in range(n)]
    
    # Calculer la précision à chaque rang
    precision_at_rank = [tp_cumul[i] / (i + 1) for i in range(n)]
    
    # Calculer le rappel à chaque rang
    recall_at_rank = [tp_cumul[i] / total_relevant for i in range(n)]
    
    # Précision globale
    precision = tp_cumul[-1] / n if n > 0 else 0
    
    # Rappel global
    recall = tp_cumul[-1] / total_relevant if total_relevant > 0 else 0
    
    # F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Average Precision (AP)
    ap = 0
    prev_recall = 0
    for i in range(n):
        if tp_at_rank[i] == 1:  # Si l'image est pertinente
            # Ajouter la précision à ce niveau de rappel
            ap += precision_at_rank[i] * (recall_at_rank[i] - prev_recall)
            prev_recall = recall_at_rank[i]
    
    # R-Precision
    # Nombre de documents pertinents dans les R premiers résultats, où R est le nombre total de documents pertinents
    r = min(total_relevant, n)
    r_precision = sum(tp_at_rank[:r]) / r if r > 0 else 0
    
    # Métriques finales
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'ap': ap,
        'map': ap,  # MAP est la moyenne des AP, mais nous n'avons qu'une seule requête
        'r_precision': r_precision,
        'precision_at_rank': precision_at_rank,
        'recall_at_rank': recall_at_rank
    }
    
    return metrics 
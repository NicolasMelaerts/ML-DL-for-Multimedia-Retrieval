from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
import os
import sys
import cv2
import numpy as np
import base64
import time
import json
import glob

# Chemin vers le dossier DESKTOP_APP
DESKTOP_APP_PATH = "/opt/DESKTOP_APP"
sys.path.append(DESKTOP_APP_PATH)

# Dossier contenant la base d'images
IMAGE_FOLDER = os.path.join(DESKTOP_APP_PATH, "MIR_DATASETS_B")

# Dossier pour les fichiers temporaires
TEMP_FOLDER = os.path.join(DESKTOP_APP_PATH, "temp")

# Dossier contenant les features
FEATURES_FOLDER = os.path.join(DESKTOP_APP_PATH, "Features")

# Importer les fonctions nécessaires
from distances import getkVoisins_deep

deep_search_bp = Blueprint('deep_search', __name__)

# Modèles disponibles
MODELS = {
    'GoogLeNet': 'GoogLeNet',
    'Inception3': 'Inception v3',
    'ResNet': 'ResNet',
    'ViT': 'Vision Transformer (ViT)',
    'VGG': 'VGG'
}

# Configuration des animaux et races pour la recherche rapide d'images
animaux = ["araignee", "chiens", "oiseaux", "poissons", "singes"]
araignees = ["barn spider", "garden spider", "orb-weaving spider", "tarantula", "trap_door spider", "wolf spider"]
chiens = ["boxer", "Chihuahua", "golden\x20retriever", "Labrador\x20retriever", "Rottweiler", "Siberian\x20husky"]
oiseaux = ["blue jay", "bulbul", "great grey owl", "parrot", "robin", "vulture"]
poissons = ["dogfish", "eagle ray", "guitarfish", "hammerhead", "ray", "tiger shark"]
singes = ["baboon", "chimpanzee", "gorilla", "macaque", "orangutan", "squirrel monkey"]

@deep_search_bp.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query_image_data = None
    metrics_data = None
    performance_info = {}
    
    if request.method == 'POST':
        # Mesurer le temps total
        total_start_time = time.time()
        
        # Récupérer les modèles sélectionnés
        selected_models = []
        if request.form.get('googlenet'): selected_models.append('GoogLeNet')
        if request.form.get('inception'): selected_models.append('Inception3')
        if request.form.get('resnet'): selected_models.append('ResNet')
        if request.form.get('vit'): selected_models.append('ViT')
        if request.form.get('vgg'): selected_models.append('VGG')
        
        if not selected_models:
            flash("Veuillez sélectionner au moins un modèle.", "warning")
            return redirect(url_for('deep_search.index'))
        
        # Récupérer le nombre de résultats à afficher
        top_k = 5
        if request.form.get('display') == 'top20':
            top_k = 20
        if request.form.get('display') == 'top50':
            top_k = 50
        
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
                
                # Extraire le nom de base de l'image (sans extension)
                query_name = os.path.splitext(os.path.basename(query_file.filename))[0]
                
                # Charger l'image pour l'affichage
                with open(temp_path, 'rb') as img_file:
                    query_image_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                image_processing_time = time.time() - image_processing_start
                performance_info["image_processing_time"] = round(image_processing_time, 4)
                
                # Mesurer le temps de chargement des features
                features_loading_start = time.time()
                
                # Charger les features et les images pour tous les modèles sélectionnés
                features_dict = {}
                image_dict = {}
                class_counts = {}
                
                for model in selected_models:
                    model_features, model_images = load_features_with_images(model, IMAGE_FOLDER)
                    if model_features:
                        features_dict[model] = model_features
                        # Fusionner les dictionnaires d'images
                        image_dict.update(model_images)
                        
                        # Compter les images par classe pour les métriques
                        update_class_counts(class_counts, model_images)
                    else:
                        flash(f"Aucune feature chargée pour {model}", "warning")
                
                features_loading_time = time.time() - features_loading_start
                performance_info["features_loading_time"] = round(features_loading_time, 4)
                performance_info["features_count"] = sum(len(f) for f in features_dict.values())
                performance_info["images_count"] = len(image_dict)
                
                # Mesurer le temps de recherche
                search_start = time.time()
                
                # Effectuer la recherche pour chaque modèle
                all_results = []
                
                for model_name, model_features in features_dict.items():
                    # Vérifier si l'image requête a des features pour ce modèle
                    # Si c'est une nouvelle image, on doit extraire ses features
                    if query_name in model_features:
                        # Rechercher les k plus proches voisins
                        neighbors = getkVoisins_deep(model_features, query_name, top_k)
                        
                        # Ajouter les résultats à la liste globale
                        for neighbor_name, dist in neighbors:
                            if neighbor_name in image_dict:
                                all_results.append((image_dict[neighbor_name], dist, model_name))
                    else:
                        flash(f"L'image requête n'a pas de features pour le modèle {model_name}", "warning")
                
                search_time = time.time() - search_start
                performance_info["search_time"] = round(search_time, 4)
                
                # Mesurer le temps de tri
                sort_start = time.time()
                
                # Trier les résultats par distance (croissante)
                all_results.sort(key=lambda x: x[1])
                
                # Limiter aux k premiers résultats
                all_results = all_results[:top_k]
                
                sort_time = time.time() - sort_start
                performance_info["sort_time"] = round(sort_time, 4)
                
                # Mesurer le temps de préparation des résultats
                display_start = time.time()
                
                # Préparer les résultats pour l'affichage
                formatted_results = []
                for i, (path, dist, model_name) in enumerate(all_results):
                    try:
                        # Charger l'image en base64
                        with open(path, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        
                        # Extraire la classe (animal/breed)
                        parts = path.split(os.sep)
                        if len(parts) >= 3:
                            # Format attendu: .../animal/race/image.jpg
                            animal_idx = max(0, len(parts) - 3)
                            breed_idx = max(0, len(parts) - 2)
                            img_class = f"{parts[animal_idx]}/{parts[breed_idx]}"
                        else:
                            img_class = "unknown"
                        
                        formatted_results.append({
                            'rank': i + 1,
                            'name': os.path.basename(path),
                            'class': img_class,
                            'distance': dist,
                            'model': model_name,
                            'image_data': img_data
                        })
                    except Exception as e:
                        flash(f"Erreur lors du traitement de l'image {path}: {str(e)}", "error")
                
                display_time = time.time() - display_start
                performance_info["display_time"] = round(display_time, 4)
                
                # Mesurer le temps de calcul des métriques
                metrics_start = time.time()
                
                # Calculer les métriques
                metrics_data = calculate_metrics(formatted_results, temp_path, class_counts)
                
                metrics_time = time.time() - metrics_start
                performance_info["metrics_time"] = round(metrics_time, 4)
                
                # Supprimer l'image temporaire
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                results = formatted_results
                
                # Temps total
                total_time = time.time() - total_start_time
                performance_info["total_time"] = round(total_time, 4)
    
    # Ajouter ces logs avant le return
    print(f"Résultats: {len(results) if results else 0}")
    print(f"Métriques disponibles: {'Oui' if metrics_data else 'Non'}")
    if performance_info:
        print(f"Temps total: {performance_info.get('total_time', 0)} secondes")
    
    return render_template(
        'deep_search.html',
        title="Moteur de Recherche par Deep Learning",
        models=MODELS,
        results=results,
        query_image=query_image_data,
        metrics=metrics_data,
        performance_info=performance_info
    )

def load_features_with_images(model_name, image_folder):
    """
    Charge les features et trouve les images correspondantes.
    
    Args:
        model_name: Nom du modèle
        image_folder: Dossier contenant les images
            
    Returns:
        Tuple (features_dict, image_dict) contenant les features et les chemins des images
    """
    start_time = time.time()
    
    feature_folder = os.path.join(FEATURES_FOLDER, model_name)
    feature_files = sorted(glob.glob(os.path.join(feature_folder, "*.txt")))
    features_dict = {}
    image_dict = {}
    
    print(f"Chargement de {len(feature_files)} fichiers depuis {feature_folder}")
    
    for file in feature_files:
        try:
            feature_vector = np.loadtxt(file, ndmin=1)
            base_name = os.path.splitext(os.path.basename(file))[0]
            
            # Stocker les features
            features_dict[base_name] = feature_vector
            
            # Trouver l'image correspondante en utilisant la méthode optimisée
            image_path = find_image_path(base_name)
            if image_path:
                image_dict[base_name] = image_path
            else:
                print(f"Aucune image trouvée pour {file} !")
        except Exception as e:
            print(f"Erreur lors du chargement de {file}: {str(e)}")
    
    elapsed_time = time.time() - start_time
    print(f"{len(features_dict)} caractéristiques chargées avec {len(image_dict)} images depuis {feature_folder} en {elapsed_time:.2f} secondes")
    
    return features_dict, image_dict

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
            direct_path = os.path.join(IMAGE_FOLDER, animal, race, f"{base_name}{img_ext}")
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
                    direct_path = os.path.join(IMAGE_FOLDER, animal, matching_race, f"{base_name}{img_ext}")
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
                    direct_path = os.path.join(IMAGE_FOLDER, animal, race, image_name)
                    if os.path.exists(direct_path):
                        return direct_path
    
    # Si on arrive ici, essayer une dernière approche
    return find_image_in_directory(IMAGE_FOLDER, base_name)

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

def find_image_in_directory(base_dir, image_name):
    """
    Recherche récursivement une image dans un dossier et ses sous-dossiers.
    
    Args:
        base_dir: Dossier de base pour la recherche
        image_name: Nom de l'image à rechercher (sans extension)
        
    Returns:
        Chemin complet de l'image si trouvée, None sinon
    """
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_name_without_ext = os.path.splitext(file)[0]
                if file_name_without_ext == image_name:
                    return os.path.join(root, file)
    return None

def update_class_counts(class_counts, image_dict):
    """Met à jour le compteur d'images par classe"""
    for image_path in image_dict.values():
        parts = image_path.split(os.sep)
        if len(parts) >= 3:
            # Format attendu: .../animal/race/image.jpg
            animal_idx = max(0, len(parts) - 3)
            breed_idx = max(0, len(parts) - 2)
            class_key = f"{parts[animal_idx]}/{parts[breed_idx]}"
            class_counts[class_key] = class_counts.get(class_key, 0) + 1
    return class_counts

def calculate_metrics(results, query_path, class_counts):
    """Calcule les métriques d'évaluation"""
    if not results:
        # Retourner des métriques vides plutôt que None
        return {
            "rappel": 0.0,
            "precision": 0.0,
            "ap": 0.0,
            "map": 0.0,
            "r_precision": 0.0,
            "precision_recall": []
        }
    
    # Extraire la classe de l'image requête
    req_class = None
    if query_path:
        parts = query_path.split(os.sep)
        if len(parts) >= 3:
            # Format attendu: .../animal/race/image.jpg
            animal_idx = max(0, len(parts) - 3)
            breed_idx = max(0, len(parts) - 2)
            req_class = f"{parts[animal_idx]}/{parts[breed_idx]}"
    
    if not req_class:
        # Essayer de déterminer la classe à partir du nom de fichier
        filename = os.path.basename(query_path)
        parts = filename.split('_')
        if len(parts) >= 2:
            req_class = f"{parts[0]}/{parts[1]}"
    
    if not req_class:
        print("Impossible de déterminer la classe de l'image requête")
        # Retourner des métriques vides plutôt que None
        return {
            "rappel": 0.0,
            "precision": 0.0,
            "ap": 0.0,
            "map": 0.0,
            "r_precision": 0.0,
            "precision_recall": []
        }
    
    # Calculer les métriques
    relevant_count = class_counts.get(req_class, 0)
    if relevant_count == 0:
        print(f"Aucune image trouvée pour la classe {req_class}")
        return {
            "rappel": 0.0,
            "precision": 0.0,
            "ap": 0.0,
            "map": 0.0,
            "r_precision": 0.0,
            "precision_recall": []
        }
    
    # Initialiser les listes pour les calculs
    relevants = []
    precisions = []
    recalls = []
    
    # Calculer précision et rappel à chaque rang
    retrieved_relevant = 0
    for result in results:
        # Vérifier si le résultat est pertinent (même classe)
        is_relevant = (result['class'] == req_class)
        relevants.append(is_relevant)
        
        if is_relevant:
            retrieved_relevant += 1
        
        # Calculer précision et rappel à ce rang
        precision = retrieved_relevant / len(relevants)
        recall = retrieved_relevant / relevant_count
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculer la précision moyenne (AP)
    ap = 0.0
    if recalls:
        # Utiliser la méthode de l'interpolation
        for i in range(11):  # 11 points: 0.0, 0.1, ..., 1.0
            r = i / 10
            # Trouver toutes les précisions à des rappels >= r
            p_at_r = [precisions[j] for j in range(len(recalls)) if recalls[j] >= r]
            if p_at_r:
                ap += max(p_at_r) / 11
    
    # Calculer R-Precision
    r_precision = 0.0
    if relevant_count <= len(relevants):
        r_precision = sum(relevants[:relevant_count]) / relevant_count
    
    # Stocker les métriques dans le dictionnaire
    metrics = {
        "rappel": recalls[-1] if recalls else 0.0,
        "precision": precisions[-1] if precisions else 0.0,
        "ap": ap,
        "map": ap,  # Pour une seule requête, MAP = AP
        "r_precision": r_precision,
        "precision_recall": list(zip(recalls, precisions))
    }
    
    print(f"Métriques calculées: Rappel={metrics['rappel']:.4f}, "
          f"Précision={metrics['precision']:.4f}, AP={metrics['ap']:.4f}, "
          f"R-Precision={metrics['r_precision']:.4f}")
    
    return metrics

@deep_search_bp.route('/metrics', methods=['POST'])
def show_metrics():
    """Endpoint pour afficher les métriques en AJAX"""
    print("Endpoint /metrics appelé")
    data = request.json
    if not data or 'metrics' not in data:
        print("Erreur: Aucune donnée de métriques fournie")
        return jsonify({'error': 'Aucune donnée de métriques fournie'}), 400
    
    metrics = data['metrics']
    print(f"Métriques reçues: {metrics}")
    
    # Vérifier que les métriques ont les propriétés nécessaires
    required_props = ['rappel', 'precision', 'ap', 'r_precision']
    missing_props = [prop for prop in required_props if prop not in metrics]
    
    if missing_props:
        print(f"Propriétés manquantes dans les métriques: {missing_props}")
        # Initialiser les propriétés manquantes à 0
        for prop in missing_props:
            metrics[prop] = 0.0
    
    # Vérifier si precision_recall existe et créer un tableau vide si ce n'est pas le cas
    if 'precision_recall' not in metrics or not metrics['precision_recall']:
        print("Données precision_recall manquantes ou vides")
        metrics['precision_recall'] = []
    
    return render_template('metrics_modal.html', metrics=metrics) 
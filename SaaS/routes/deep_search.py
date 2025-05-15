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
    'Inception': 'Inception v3',
    'ResNet': 'ResNet',
    'ViT': 'Vision Transformer (ViT)',
    'VGG': 'VGG'
}

@deep_search_bp.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query_image_data = None
    metrics_data = None
    
    if request.method == 'POST':
        # Récupérer les modèles sélectionnés
        selected_models = []
        if request.form.get('googlenet'): selected_models.append('GoogLeNet')
        if request.form.get('inception'): selected_models.append('Inception')
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
                
                # Trier les résultats par distance (croissante)
                all_results.sort(key=lambda x: x[1])
                
                # Limiter aux k premiers résultats
                all_results = all_results[:top_k]
                
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
                
                # Calculer les métriques
                metrics_data = calculate_metrics(formatted_results, temp_path, class_counts)
                
                # Supprimer l'image temporaire
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                results = formatted_results
    
    # Ajouter ces logs avant le return
    print(f"Résultats: {len(results) if results else 0}")
    print(f"Métriques disponibles: {'Oui' if metrics_data else 'Non'}")
    
    return render_template(
        'deep_search.html',
        title="Moteur de Recherche par Deep Learning",
        models=MODELS,
        results=results,
        query_image=query_image_data,
        metrics=metrics_data
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
            
            # Trouver l'image correspondante
            # D'abord, essayer de trouver l'image directement
            image_path_jpg = os.path.join(image_folder, base_name + ".jpg")
            image_path_jpeg = os.path.join(image_folder, base_name + ".jpeg")
            image_path_png = os.path.join(image_folder, base_name + ".png")
            
            # Vérifier l'existence des images
            if os.path.exists(image_path_jpg):
                image_dict[base_name] = image_path_jpg
            elif os.path.exists(image_path_jpeg):
                image_dict[base_name] = image_path_jpeg
            elif os.path.exists(image_path_png):
                image_dict[base_name] = image_path_png
            else:
                # Si l'image n'est pas trouvée directement, essayer de la chercher dans les sous-dossiers
                found_path = find_image_in_directory(image_folder, base_name)
                if found_path:
                    image_dict[base_name] = found_path
                else:
                    print(f"Aucune image trouvée pour {file} !")
        except Exception as e:
            print(f"Erreur lors du chargement de {file}: {str(e)}")
    
    print(f"{len(features_dict)} caractéristiques chargées avec {len(image_dict)} images depuis {feature_folder}")
    
    return features_dict, image_dict

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
    data = request.json
    if not data or 'metrics' not in data:
        return jsonify({'error': 'Aucune donnée de métriques fournie'}), 400
    
    metrics = data['metrics']
    return render_template('metrics_modal.html', metrics=metrics) 
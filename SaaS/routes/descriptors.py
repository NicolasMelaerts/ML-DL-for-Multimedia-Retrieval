from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
import os
import sys
import time
import threading

# Importer les fonctions de calcul de descripteurs
sys.path.append('DESKTOP_APP')
from descriptors import (
    generateHistogramme_Color, 
    generateHistogramme_HSV, 
    generateSIFT, 
    generateORB,
    generateGLCM,
    generateLBP,
    generateHOG
)

descriptors_bp = Blueprint('descriptors', __name__)

# Dictionnaire des descripteurs avec leurs descriptions
descriptors_info = {
    "SIFT": "Scale-Invariant Feature Transform (SIFT)\n\n"
           "• Détecte et décrit les points d'intérêt locaux\n"
           "• Invariant à l'échelle et à la rotation\n"
           "• Partiellement invariant aux changements d'illumination\n"
           "• Utile pour la reconnaissance d'objets",
           
    "ORB": "Oriented FAST and Rotated BRIEF (ORB)\n\n"
           "• Alternative rapide à SIFT et SURF\n"
           "• Combine le détecteur FAST et le descripteur BRIEF\n"
           "• Gère l'orientation pour plus de robustesse\n"
           "• Efficace en termes de calcul",
           
    "Hist Couleur": "Histogramme de Couleur BGR\n\n"
                   "• Capture la distribution des couleurs (bleu, vert, rouge)\n"
                   "• Simple mais efficace pour décrire le contenu global\n"
                   "• Sensible aux changements d'illumination\n"
                   "• Ignore la disposition spatiale des couleurs",
                   
    "Hist HSV": "Histogramme HSV\n\n"
                "• Représente les couleurs en Teinte-Saturation-Valeur\n"
                "• Plus proche de la perception humaine des couleurs\n"
                "• Meilleure robustesse aux changements d'illumination\n"
                "• Utile pour la segmentation basée sur la couleur",
                
    "GLCM": "Matrice de Co-occurrence des Niveaux de Gris (GLCM)\n\n"
            "• Analyse la texture en considérant les relations spatiales\n"
            "• Calcule la fréquence d'apparition de paires de pixels\n"
            "• Extrait des propriétés comme contraste, homogénéité, énergie\n"
            "• Efficace pour distinguer différentes textures",
            
    "LBP": "Motif Binaire Local (LBP)\n\n"
           "• Descripteur de texture robuste\n"
           "• Étiquette les pixels en seuillant leur voisinage\n"
           "• Invariant aux changements monotones d'illumination\n"
           "• Utilisé pour la reconnaissance faciale et l'analyse de texture",
           
    "HOG": "Histogramme des Gradients Orientés (HOG)\n\n"
           "• Compte les occurrences des orientations de gradient\n"
           "• Divise l'image en cellules et blocs\n"
           "• Robuste aux changements d'illumination et aux petites déformations\n"
           "• Très utilisé pour la détection d'objets et de personnes"
}

# Variable globale pour stocker l'état du calcul en cours
current_progress = {
    'descriptor': 'Aucun',
    'progress': 0,
    'is_running': False
}

def progress_callback(progress_value):
    """Fonction de callback pour mettre à jour la progression"""
    current_progress['progress'] = progress_value

@descriptors_bp.route('/progress')
def get_progress():
    """Route API pour récupérer la progression actuelle"""
    return jsonify(current_progress)

@descriptors_bp.route('/', methods=['GET', 'POST'])
def index():
    # Vérifier quels descripteurs sont déjà calculés
    descriptor_status = check_descriptor_status()
    
    if request.method == 'POST':
        # Récupérer le dossier d'images
        dataset_dir = request.form.get('dataset_dir', 'DESKTOP_APP/MIR_DATASETS_B')
        
        # Vérifier si le dossier existe
        if not os.path.exists(dataset_dir):
            flash(f"Le dossier {dataset_dir} n'existe pas.", "error")
            return redirect(url_for('descriptors.index'))
        
        # Récupérer les descripteurs sélectionnés
        selected_descriptors = []
        if request.form.get('hist_color'): selected_descriptors.append('BGR')
        if request.form.get('hist_hsv'): selected_descriptors.append('HSV')
        if request.form.get('sift'): selected_descriptors.append('SIFT')
        if request.form.get('orb'): selected_descriptors.append('ORB')
        if request.form.get('glcm'): selected_descriptors.append('GLCM')
        if request.form.get('lbp'): selected_descriptors.append('LBP')
        if request.form.get('hog'): selected_descriptors.append('HOG')
        
        if not selected_descriptors:
            flash("Veuillez sélectionner au moins un descripteur.", "warning")
            return redirect(url_for('descriptors.index'))
        
        # Vérifier si un calcul est déjà en cours
        if current_progress['is_running']:
            flash("Un calcul est déjà en cours. Veuillez attendre qu'il se termine.", "warning")
            return redirect(url_for('descriptors.index'))
        
        # Lancer le calcul des descripteurs dans un thread séparé
        thread = threading.Thread(target=calculate_descriptors, args=(dataset_dir, selected_descriptors))
        thread.daemon = True
        thread.start()
        
        flash("Calcul des descripteurs lancé en arrière-plan.", "info")
        return redirect(url_for('descriptors.index'))
    
    return render_template(
        'descriptors.html', 
        title="Calcul des Descripteurs",
        descriptors_info=descriptors_info,
        descriptor_status=descriptor_status,
        current_progress=current_progress
    )

def calculate_descriptors(dataset_dir, selected_descriptors):
    """Fonction pour calculer les descripteurs sélectionnés"""
    global current_progress
    current_progress['is_running'] = True
    
    # Créer le dossier DESKTOP_APP/Descripteurs s'il n'existe pas
    descriptors_dir = "DESKTOP_APP/Descripteurs"
    if not os.path.exists(descriptors_dir):
        os.makedirs(descriptors_dir)
    
    # Changer le répertoire de travail pour que les descripteurs soient créés au bon endroit
    original_dir = os.getcwd()
    os.chdir("DESKTOP_APP")
    
    results = []
    descriptor_status = check_descriptor_status()
    
    try:
        for desc in selected_descriptors:
            # Vérifier si le descripteur est déjà calculé
            if descriptor_status.get(desc, False):
                results.append(f"Descripteur {desc} déjà calculé, calcul ignoré")
                continue
                
            # Mettre à jour le descripteur en cours
            current_progress['descriptor'] = desc
            current_progress['progress'] = 0
            
            # Calculer le descripteur
            start_time = time.time()
            try:
                if desc == 'BGR':
                    generateHistogramme_Color(dataset_dir.replace('DESKTOP_APP/', ''), progress_callback=progress_callback)
                elif desc == 'HSV':
                    generateHistogramme_HSV(dataset_dir.replace('DESKTOP_APP/', ''), progress_callback=progress_callback)
                elif desc == 'SIFT':
                    generateSIFT(dataset_dir.replace('DESKTOP_APP/', ''), progress_callback=progress_callback)
                elif desc == 'ORB':
                    generateORB(dataset_dir.replace('DESKTOP_APP/', ''), progress_callback=progress_callback)
                elif desc == 'GLCM':
                    generateGLCM(dataset_dir.replace('DESKTOP_APP/', ''), progress_callback=progress_callback)
                elif desc == 'LBP':
                    generateLBP(dataset_dir.replace('DESKTOP_APP/', ''), progress_callback=progress_callback)
                elif desc == 'HOG':
                    generateHOG(dataset_dir.replace('DESKTOP_APP/', ''), progress_callback=progress_callback)
                
                elapsed_time = time.time() - start_time
                results.append(f"Descripteur {desc} calculé en {elapsed_time:.2f} secondes")
            except Exception as e:
                results.append(f"Erreur lors du calcul du descripteur {desc}: {str(e)}")
    finally:
        # Restaurer le répertoire de travail
        os.chdir(original_dir)
        
        # Réinitialiser la progression
        current_progress['descriptor'] = 'Terminé'
        current_progress['progress'] = 100
        current_progress['is_running'] = False

def check_descriptor_status():
    """Vérifie quels descripteurs sont déjà calculés"""
    status = {
        'BGR': False,
        'HSV': False,
        'SIFT': False,
        'ORB': False,
        'GLCM': False,
        'LBP': False,
        'HOG': False
    }
    
    # Vérifier si le dossier Descripteurs existe
    descriptors_dir = "DESKTOP_APP/Descripteurs"
    if not os.path.exists(descriptors_dir):
        return status
    
    # Vérifier chaque descripteur
    for desc in status.keys():
        desc_path = os.path.join(descriptors_dir, desc)
        if os.path.exists(desc_path) and os.path.isdir(desc_path):
            # Vérifier s'il y a des fichiers dans le dossier
            files = [f for f in os.listdir(desc_path) if f.endswith('.txt')]
            if files:
                status[desc] = True
    
    return status 
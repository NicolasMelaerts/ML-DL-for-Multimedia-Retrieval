from flask import Blueprint, render_template, request, send_file
import os
import base64

display_bp = Blueprint('display', __name__)

# Chemin vers le dossier contenant les images
DATASETS_DIR = "/opt/DESKTOP_APP/MIR_DATASETS_B"

# Listes des animaux et races
animaux = ["araignee", "chiens", "oiseaux", "poissons", "singes"]
araignees = ["barn spider", "garden spider", "orb-weaving spider", "tarantula", "trap_door spider", "wolf spider"]
chiens = ["boxer", "Chihuahua", "golden retriever", "Labrador retriever", "Rottweiler", "Siberian husky"]
oiseaux = ["blue jay", "bulbul", "great grey owl", "parrot", "robin", "vulture"]
poissons = ["dogfish", "eagle ray", "guitarfish", "hammerhead", "ray", "tiger shark"]
singes = ["baboon", "chimpanzee", "gorilla", "macaque", "orangutan", "squirrel monkey"]

def get_all_images():
    """Parcourt le dossier DATASETS_DIR et retourne la liste de tous les chemins d'images .jpg"""
    image_paths = []
    try:
        for animal in animaux:
            races = []
            if animal == "araignee": races = araignees
            elif animal == "chiens": races = chiens
            elif animal == "oiseaux": races = oiseaux
            elif animal == "poissons": races = poissons
            elif animal == "singes": races = singes
            
            for race in races:
                race_dir = os.path.join(DATASETS_DIR, animal, race)
                if os.path.isdir(race_dir):
                    for fname in os.listdir(race_dir):
                        if fname.lower().endswith('.jpg'):
                            image_paths.append(os.path.join(animal, race, fname))
    except Exception as e:
        print(f"Erreur lors de la recherche d'images: {e}")
    return image_paths

def get_image_base64(image_path):
    """Convertit une image en base64 pour l'affichage HTML"""
    try:
        with open(os.path.join(DATASETS_DIR, image_path), "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Erreur lors de la conversion de l'image en base64: {e}")
        return None

@display_bp.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    images = get_all_images()
    
    if not images:
        error_message = "Aucune image trouvée dans le dossier spécifié."
    
    current_index = 0
    if request.method == 'POST':
        action = request.form.get('action')
        current_index = int(request.form.get('current_index', 0))
        
        if action == 'prev':
            current_index = max(0, current_index - 1)
        elif action == 'next':
            current_index = min(len(images) - 1, current_index + 1)
        elif action == 'select':
            current_index = int(request.form.get('image_select', 0))
    
    selected_image = images[current_index] if images else None
    image_data = get_image_base64(selected_image) if selected_image else None
    
    return render_template(
        'display.html',
        images=images,
        current_index=current_index,
        selected_image=selected_image,
        image_data=image_data,
        total=len(images) if images else 0,
        title="Affichage des Images",
        error=error_message
    )

@display_bp.route('/image/<path:image_path>')
def serve_image(image_path):
    """Sert une image depuis le dossier DATASETS_DIR"""
    try:
        full_path = os.path.join(DATASETS_DIR, image_path)
        return send_file(full_path)
    except Exception as e:
        return f"Erreur: Impossible de charger l'image. {str(e)}" 
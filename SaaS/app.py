from flask import Flask, render_template, request
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

# Configuration
MODEL_PATH = "../DESKTOP_APP/Transformer/sentence_transformer_model"
CAPTIONS_FILE = "../DESKTOP_APP/Transformer/captions.json"
EMBEDDINGS_DIR = "../DESKTOP_APP/Transformer/embeddings_output"
DATASETS_DIR = "../DESKTOP_APP/MIR_DATASETS_B"

# Variables globales
model = None
captions = {}

@app.route('/', methods=['GET', 'POST'])
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
    
    return render_template('index.html', 
                          results=results, 
                          error_message=error_message, 
                          success_message=success_message,
                          query=query,
                          top_k=top_k)

if __name__ == '__main__':
    # Créer le dossier templates s'il n'existe pas
    os.makedirs('templates', exist_ok=True)
    
    # Créer le template HTML s'il n'existe pas
    template_path = os.path.join('templates', 'index.html')
    if not os.path.exists(template_path):
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write('''<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recherche d'Images par Texte</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .result-card {
            margin-bottom: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .result-image {
            max-width: 200px;
            max-height: 200px;
            object-fit: contain;
        }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1 class="text-center mb-4">Recherche d'Images par Texte</h1>
        
        <div class="card mb-4">
            <div class="card-header">
                <h5>Recherche</h5>
            </div>
            <div class="card-body">
                <form method="post" action="">
                    <div class="mb-3">
                        <label for="query" class="form-label">Description textuelle:</label>
                        <input type="text" class="form-control" id="query" name="query" 
                               placeholder="Entrez une description d'image (ex: 'a bird standing on the ground')"
                               value="{{ query }}" required>
                    </div>
                    
                    <div class="mb-3">
                        <label for="top_k" class="form-label">Nombre de résultats:</label>
                        <input type="number" class="form-control" id="top_k" name="top_k" 
                               min="1" max="20" value="{{ top_k }}">
                    </div>
                    
                    <button type="submit" class="btn btn-primary">Rechercher</button>
                </form>
            </div>
        </div>
        
        {% if error_message %}
            <div class="alert alert-danger" role="alert">
                {{ error_message }}
            </div>
        {% endif %}
        
        {% if success_message %}
            <div class="alert alert-success" role="alert">
                {{ success_message }}
            </div>
        {% endif %}
        
        {% if results %}
            <h2>Résultats</h2>
            
            {% for result in results %}
                <div class="result-card">
                    <div class="row">
                        <div class="col-md-3">
                            {% if result.image_data %}
                                <img src="data:image/jpeg;base64,{{ result.image_data }}" class="result-image" alt="Image">
                            {% else %}
                                <div class="alert alert-warning">Image non trouvée</div>
                            {% endif %}
                        </div>
                        <div class="col-md-9">
                            <p><strong>Chemin:</strong> {{ result.image_path }}</p>
                            <p><strong>Description:</strong> {{ result.caption }}</p>
                            <p><strong>Score de similarité:</strong> {{ "%.4f"|format(result.similarity) }}</p>
                            <p><strong>Animal:</strong> {{ result.animal }}</p>
                            <p><strong>Race:</strong> {{ result.race }}</p>
                        </div>
                    </div>
                </div>
            {% endfor %}
        {% elif request.method == 'POST' and not error_message %}
            <div class="alert alert-info" role="alert">
                Aucun résultat trouvé.
            </div>
        {% endif %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>''')
    
    # Lancer l'application
    app.run(host='0.0.0.0', port=5000, debug=True)

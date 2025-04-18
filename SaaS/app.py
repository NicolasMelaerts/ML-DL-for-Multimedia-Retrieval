# -*- coding: utf-8 -*-
"""
Point d'entrée principal de l'application Flask
"""

from flask import Flask, render_template, redirect, url_for, request, session, flash
import os
from werkzeug.utils import secure_filename
import secrets

# Créer le répertoire routes s'il n'existe pas
os.makedirs('routes', exist_ok=True)
# Créer un fichier __init__.py vide s'il n'existe pas
if not os.path.exists('routes/__init__.py'):
    with open('routes/__init__.py', 'w') as f:
        pass

# Importer les routes pour chaque fonctionnalité
try:
    from routes.home import home_bp
    from routes.text_search import text_search_bp
    # Commentez les imports qui ne sont pas encore implémentés
    # from routes.descriptors import descriptors_bp
    # from routes.display import display_bp
    # from routes.search import search_bp
    # from routes.deep_search import deep_search_bp
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    # Créer des blueprints vides pour les modules manquants
    from flask import Blueprint
    if 'home_bp' not in locals():
        home_bp = Blueprint('home', __name__)
        @home_bp.route('/')
        def index():
            return "Page d'accueil temporaire"
    
    if 'text_search_bp' not in locals():
        text_search_bp = Blueprint('text_search', __name__)
        @text_search_bp.route('/')
        def index():
            return "Page de recherche par texte temporaire"
    
    # Créez des blueprints vides pour les autres modules si nécessaire

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration de l'application
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Créer le dossier d'upload s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Enregistrer les blueprints
app.register_blueprint(home_bp, url_prefix='/')
app.register_blueprint(text_search_bp, url_prefix='/text-search')

# Commentez les blueprints qui ne sont pas encore implémentés
# app.register_blueprint(descriptors_bp, url_prefix='/descriptors')
# app.register_blueprint(display_bp, url_prefix='/display')
# app.register_blueprint(search_bp, url_prefix='/search')
# app.register_blueprint(deep_search_bp, url_prefix='/deep-search')

# Fonction pour vérifier les extensions de fichier autorisées
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.errorhandler(404)
def page_not_found(e):
    return "Page non trouvée", 404

@app.errorhandler(500)
def server_error(e):
    return "Erreur serveur", 500

if __name__ == "__main__":
    print("Démarrage de l'application Flask...")
    app.run(host='0.0.0.0', port=5000, debug=True)
# -*- coding: utf-8 -*-
"""
Point d'entrée principal de l'application Flask
"""

from flask import Flask, render_template, redirect, url_for, request, session, flash
import os
from werkzeug.utils import secure_filename
import secrets

# Importer les routes pour chaque fonctionnalité
from routes.home import home_bp
from routes.descriptors import descriptors_bp
from routes.display import display_bp
from routes.search import search_bp
from routes.text_search import text_search_bp
from routes.deep_search import deep_search_bp

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
app.register_blueprint(descriptors_bp, url_prefix='/descriptors')
app.register_blueprint(display_bp, url_prefix='/display')
app.register_blueprint(search_bp, url_prefix='/search')
app.register_blueprint(text_search_bp, url_prefix='/text-search')
app.register_blueprint(deep_search_bp, url_prefix='/deep-search')

# Fonction pour vérifier les extensions de fichier autorisées
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == "__main__":
    # Charger et appliquer le style CSS (déjà géré par Flask/templates)
    print("Démarrage de l'application Flask...")
    app.run(debug=True)
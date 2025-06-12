# -*- coding: utf-8 -*-
"""
Application principale Flask
"""

import os
import sys
from functools import wraps
from flask import Flask, render_template, redirect, url_for, request, jsonify, send_from_directory, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Ajouter le chemin DESKTOP_APP au PYTHONPATH
desktop_app_path = '/opt/DESKTOP_APP'
if desktop_app_path not in sys.path:
    sys.path.append(desktop_app_path)

# Maintenant vous pouvez importer les modules de DESKTOP_APP
from routes.text_search import text_search_bp
from routes.descriptors import descriptors_bp
from routes.display import display_bp
from routes.search import search_bp
from routes.deep_search import deep_search_bp
import requests

app = Flask(__name__)

# Définir une clé secrète pour l'application
app.config['SECRET_KEY'] = 'une_cle_secrete_tres_difficile_a_deviner'

# Configuration Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Classe utilisateur simple pour l'authentification
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password_hash = password

# Utilisateur unique (à remplacer par une base de données dans un environnement de production)
users = {
    'admin': User(1, 'admin', generate_password_hash('password123'))
}

@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if user.id == int(user_id):
            return user
    return None

# Routes d'authentification
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users and check_password_hash(users[username].password_hash, password):
            login_user(users[username])
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash('Identifiants incorrects', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Décorateur personnalisé pour exclure le service Vasarely de l'authentification
def login_required_except_vasarely(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.endpoint in ['vasarely', 'generate_vasarely'] or request.path.startswith('/vasarely'):
            return f(*args, **kwargs)
        return login_required(f)(*args, **kwargs)
    return decorated_function

# Appliquer le décorateur personnalisé à toutes les routes sauf login et vasarely
@app.before_request
def before_request():
    if request.endpoint in ['login', 'logout', 'vasarely', 'generate_vasarely', 'static'] or request.path.startswith('/vasarely'):
        return
    
    if not current_user.is_authenticated:
        return redirect(url_for('login', next=request.url))

# Enregistrement des blueprints
app.register_blueprint(text_search_bp, url_prefix='/text_search')
app.register_blueprint(descriptors_bp, url_prefix='/descriptors')
app.register_blueprint(display_bp, url_prefix='/display')
app.register_blueprint(search_bp, url_prefix='/search')
app.register_blueprint(deep_search_bp, url_prefix='/deep_search')

@app.route('/')
def home():
    """
    Page d'accueil de l'application
    """
    return render_template('home.html', title="Système de Recherche d'Images")

# Routes de redirection vers les différentes pages
@app.route('/display_page')
def display_page():
    """
    Redirection vers la page d'affichage des images
    """
    return redirect(url_for('display.index'))

@app.route('/descriptors_page')
def descriptors_page():
    """
    Redirection vers la page de calcul des descripteurs
    """
    return redirect(url_for('descriptors.index'))

@app.route('/search_page')
def search_page():
    """
    Redirection vers la page de recherche avec descripteurs
    """
    return redirect(url_for('search.index'))

@app.route('/text_search_page')
def text_search_page():
    """
    Redirection vers la page de recherche d'images par texte
    """
    return redirect(url_for('text_search.index'))

@app.route('/deep_search_page')
def deep_search_page():
    """
    Redirection vers la page de recherche par deep learning
    """
    return redirect(url_for('deep_search.index'))

@app.route('/vasarely')
def vasarely():
    """
    Page pour générer des pavages d'hexagones style Vasarely
    """
    return render_template('vasarely.html', title="Pavage d'Hexagones Style Vasarely")

@app.route('/api/generate_vasarely', methods=['POST'])
def generate_vasarely():
    """
    API pour générer un pavage d'hexagones style Vasarely
    """
    # Récupérer les données JSON de la requête
    data = request.json
    
    # URL du service vasarely
    vasarely_url = os.environ.get('VASARELY_SERVICE_URL', 'http://vasarely:5002')
    
    try:
        # Ajoutez un timeout plus long pour les requêtes
        response = requests.post(
            f"{vasarely_url}/generate", 
            json=data,
            timeout=120  # Timeout de 2 minutes
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            app.logger.error(f"Erreur du service vasarely: {response.text}")
            return jsonify({"error": f"Erreur lors de la génération du pavage: {response.text}"}), 500
    
    except requests.exceptions.ConnectionError as e:
        app.logger.error(f"Erreur de connexion au service vasarely: {str(e)}")
        return jsonify({
            "error": "Impossible de se connecter au service vasarely. Le service est peut-être indisponible ou surchargé."
        }), 503
    
    except requests.exceptions.Timeout as e:
        app.logger.error(f"Timeout lors de la connexion au service vasarely: {str(e)}")
        return jsonify({
            "error": "Le service vasarely a mis trop de temps à répondre. Veuillez réessayer plus tard."
        }), 504
    
    except Exception as e:
        app.logger.error(f"Erreur inattendue: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
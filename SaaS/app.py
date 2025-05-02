# -*- coding: utf-8 -*-
"""
Application principale Flask
"""

from flask import Flask, render_template, redirect, url_for
from routes.text_search import text_search_bp
from routes.descriptors import descriptors_bp
from routes.display import display_bp
from routes.search import search_bp
from routes.deep_search import deep_search_bp

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
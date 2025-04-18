# -*- coding: utf-8 -*-
"""
Routes pour la page d'accueil de l'application
"""

from flask import Blueprint, render_template, redirect, url_for, request, session, flash

home_bp = Blueprint('home', __name__)

@home_bp.route('/')
def index():
    """Affiche la page d'accueil avec les différentes options"""
    return render_template('home.html', title="Système de Recherche d'Images")

# Vous pouvez ajouter d'autres routes liées à la page d'accueil ici si nécessaire
from flask import Blueprint, render_template

search_bp = Blueprint('search', __name__)

@search_bp.route('/')
def index():
    return render_template('search.html', title="Moteur de Recherche avec descripteurs") 
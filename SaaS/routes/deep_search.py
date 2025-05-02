from flask import Blueprint, render_template

deep_search_bp = Blueprint('deep_search', __name__)

@deep_search_bp.route('/')
def index():
    return render_template('deep_search.html', title="Moteur de Recherche par Deep Learning") 
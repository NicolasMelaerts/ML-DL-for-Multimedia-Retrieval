from flask import Blueprint, render_template

descriptors_bp = Blueprint('descriptors', __name__)

@descriptors_bp.route('/')
def index():
    return render_template('descriptors.html', title="Calcul des Descripteurs") 
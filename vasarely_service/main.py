from flask import Flask, request, jsonify
import base64
from io import BytesIO
import turtle
from math import pi, sin, cos, sqrt, acos, asin, atan2
import os
import tempfile
import time
import logging
import traceback

app = Flask(__name__)

# Configurez le logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deformation(p, centre, rayon):
    """ Calcul des coordonnées d'un point suite à la déformation engendrée par la sphère émergeante
        Entrées :
          p : coordonnées (x, y, z) du point du dalage à tracer (z = 0) AVANT déformation
          centre : coordonnées (X0, Y0, Z0) du centre de la sphère
          rayon : rayon de la sphère
        Sorties : coordonnées (xprim, yprim, zprim) du point du dallage à tracer APRÈS déformation
    """
    x, y, z = p
    xprim, yprim, zprim = x, y, z
    xc, yc, zc = centre
    if rayon**2 > zc**2:
        zc = zc if zc <= 0 else -zc
        r = sqrt(
            (x - xc) ** 2 + (y - yc) ** 2)  # distance horizontale depuis le point à dessiner jusqu'à l'axe de la sphère
        rayon_emerge = sqrt(rayon ** 2 - zc ** 2)           # rayon de la partie émergée de la sphère
        rprim = rayon * sin(acos(-zc / rayon) * r / rayon_emerge) if rayon_emerge > 0 else 0
        if 0 < r <= rayon_emerge:                 # calcul de la déformation dans les autres cas
            xprim = xc + (x - xc) * rprim / r           # les nouvelles coordonnées sont proportionnelles aux anciennes
            yprim = yc + (y - yc) * rprim / r
        if r <= rayon_emerge:
            beta = asin(rprim / rayon)
            zprim = zc + rayon * cos(beta)
            if centre[2] > 0:
                zprim = -zprim
    return (xprim, yprim, zprim)

def hexagone(t, p, longueur, col, centre, rayon):
    """
    Fonction qui trace un hexagone
    Entrées :
     t: objet turtle
     p: tuple de trois composantes donnant la valeur des trois coordonnées, du point avant déformation
     où l'hexagone doit être peint
     longueur: distance entre le centre et n'importe quel point de l'hexagone
     col: tuple contenant les trois couleurs qui vont être utilisées pour dessiner les hexagones
     centre: tuple de trois composantes donnant les coordonnées du centre de la sphère de déformation
     rayon: rayon de la sphère de déformation
    Sorties :
     Si les coordonnées (p) de l'hexagone qui doit être peint se trouve dans la sphère de déformation, il sera déformé
     sinon ce sera un hexagone régulier
    """
    (x, y, z) = p
    (pprim_x, pprim_y, pprim_z) = deformation((x, y, z), centre, rayon)
    (col1, col2, col3) = col
    angle = -pi/3
    t.up()
    t.goto(pprim_x, pprim_y)       # coordonnées du centre de l'hexagone
    losange=0
    while losange != 3:                 # un losange qui pivote pour former un hexagone
        angle += pi/3
        t.color(col1)
        t.begin_fill()
        for i in range(3):          # pour chaques losanges à partir du centre de l'hexagone
            x_d, y_d, z_d = deformation((x+longueur*cos(angle), y+longueur*sin(angle), z), centre, rayon)
            # calcule nouvelles coordonnées des points
            t.goto(x_d, y_d)
            angle += pi/3
        t.goto(pprim_x, pprim_y)
        t.end_fill()
        col1 = col3                     # changement de couleur du losange
        col3 = col2
        losange += 1
    t.up()

def pavage(t, inf_gauche, sup_droit, longueur, col, centre, rayon):
    """
    Réalisation du pavage d'hexagone avec une sphère de déformation
    Entrées :
     t: objet turtle
     inf_gauche: coordonnées du coin inférieur gauche
     sup_droit: coordonnées du coin supérieur droit
     longueur: distance entre le centre et n'importe quel point de l'hexagone
     col: tuple contenant les trois couleurs qui vont être utilisées pour dessiner les hexagones
     centre: tuple de trois composantes donnant les coordonnées du centre de la sphère de déformation
     rayon: rayon de la sphère de déformation
    Sortie : pavage d'hexagones avec des hexagones déformés dans la sphère de déformation
    """
    (x, y, z) = (inf_gauche, inf_gauche, 0)
    y = y-(longueur*sin(pi/3))
    while y <= sup_droit:       # le pavage ne continue pas au-delà de sup_droit
        y = y+(longueur*sin(pi/3))                # la distance entre deux lignes
        x = inf_gauche
        while x <= (sup_droit+longueur):            # première ligne d'hexagones qui ne dépasse pas sup_droit
            p = (x, y, 0)                          # nouvelles coordonnées des points où doit être peint l'hexagone
            hexagone(t, p, longueur, col, centre, rayon)
            x = x+(3*longueur)                        # distance entre deux hexagones d'une même ligne
        y = y+(longueur*sin(pi/3))
        x = inf_gauche
        x = x + (1.5 * longueur) - (3*longueur)
        # deuxième ligne d'hexagone décalée de 1.5*longueur par rapport à la première ligne
        while x <= (sup_droit-(3*longueur)):       # ne dépasse pas sup_droit
            x = x+(3*longueur)
            p = (x, y, 0)
            hexagone(t, p, longueur, col, centre, rayon)

def generate_pavage(inf_gauche, sup_droit, longueur, colors, centre_x, centre_y, centre_z, rayon):
    logger.info(f"Génération du pavage avec les paramètres: inf_gauche={inf_gauche}, sup_droit={sup_droit}, longueur={longueur}")
    
    # Créer un fichier temporaire avec un nom unique
    import uuid
    unique_id = str(uuid.uuid4())
    tmp_path = f"/tmp/pavage_{unique_id}.eps"
    
    try:
        # Configurer turtle pour dessiner dans un canvas hors écran
        screen = turtle.Screen()
        screen.reset()  # Réinitialiser l'écran pour éviter les problèmes avec les générations multiples
        screen.setup(800, 800)
        screen.tracer(0)  # Désactiver l'animation pour accélérer le dessin
        
        t = turtle.Turtle()
        t.hideturtle()
        t.speed(0)  # Vitesse maximale
        
        # Ajouter plus de logging
        logger.info("Initialisation de turtle...")
        
        # Dessiner le pavage
        pavage(t, inf_gauche, sup_droit, longueur, colors, (centre_x, centre_y, centre_z), rayon)
        
        # Mettre à jour l'écran une seule fois à la fin
        screen.update()
        
        # Sauvegarder l'image
        canvas = screen.getcanvas()
        canvas.postscript(file=tmp_path)
        
        # Convertir EPS en PNG avec PIL
        try:
            from PIL import Image, ImageDraw
            import subprocess
            
            # Utiliser ghostscript pour convertir EPS en PNG
            png_path = tmp_path.replace('.eps', '.png')
            subprocess.run(['gs', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=png16m', 
                            f'-sOutputFile={png_path}', '-r300', tmp_path], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Lire l'image PNG
            with open(png_path, 'rb') as img_file:
                img_data = img_file.read()
            
            # Nettoyer les fichiers temporaires
            os.remove(tmp_path)
            os.remove(png_path)
            
            # Fermer turtle proprement
            screen.clear()
            screen.bye()
            
            # Encoder en base64
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            logger.info("Image générée avec succès")
            return f"data:image/png;base64,{img_base64}"
        
        except Exception as e:
            # Si la conversion échoue, retourner l'erreur
            logger.error(f"Erreur lors de la conversion de l'image: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Erreur: {str(e)}"
    except Exception as e:
        logger.error(f"Erreur lors de la génération du pavage: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Erreur: {str(e)}"
    finally:
        # S'assurer que turtle est fermé
        try:
            turtle.TurtleScreen._RUNNING = True
            screen.clear()
            screen.bye()
            logger.info("Turtle fermé avec succès")
        except Exception as cleanup_error:
            logger.error(f"Erreur lors du nettoyage de turtle: {str(cleanup_error)}")

@app.route('/generate', methods=['POST'])
def generate_hexagone_pavage():
    logger.info("Requête reçue pour générer un pavage")
    
    # Récupérer les paramètres
    data = request.json
    logger.info(f"Paramètres reçus: {data}")
    inf_gauche = int(data.get('inf_gauche', -305))
    sup_droit = int(data.get('sup_droit', 305))
    longueur = int(data.get('longueur', 50))
    
    # Récupérer les couleurs
    color1 = data.get('color1', 'lime')
    color2 = data.get('color2', 'black')
    color3 = data.get('color3', 'blue')
    colors = (color1, color2, color3)
    
    # Récupérer les paramètres de la sphère
    centre_x = int(data.get('centre_x', -50))
    centre_y = int(data.get('centre_y', -50))
    centre_z = int(data.get('centre_z', 0))
    rayon = int(data.get('rayon', 300))
    
    try:
        # Générer l'image
        logger.info("Début de la génération de l'image")
        image_data = generate_pavage(inf_gauche, sup_droit, longueur, colors, centre_x, centre_y, centre_z, rayon)
        logger.info("Image générée avec succès")
        
        return jsonify({
            'image_url': image_data
        })
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/fonts', methods=['GET'])
def list_fonts():
    # Cette route est maintenue pour compatibilité
    return jsonify({'fonts': ["standard"]})

# À la fin du fichier, ajoutez un gestionnaire d'erreurs global
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Erreur non gérée: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Démarrage du service vasarely sur le port 5002")
    app.run(host='0.0.0.0', port=5002)

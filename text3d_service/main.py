from flask import Flask, request, jsonify
import pyfiglet

app = Flask(__name__)

@app.route("/ascii-text")
def ascii_text():
    text = request.args.get("text", "Merci de votre ecoute \n Par Nicolas Melaerts et Kenza Khemar")
    ascii_art = pyfiglet.figlet_format(text)
    return f"<pre style='font-family: monospace; font-size: 12px;'>{ascii_art}</pre>"

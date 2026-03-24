import os, json
from flask import Flask, render_template, request
from utils.segmentation_logic import process_image_segmentation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'

# Stockage temporaire des résultats pour l'onglet Régions
results_cache = {"selected": None, "regions": []}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        img_name = request.form.get('img')
        path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        # On traite l'image
        name, regions = process_image_segmentation(path, app.config['PROCESSED_FOLDER'])
        results_cache["selected"] = name
        results_cache["regions"] = regions
        return render_template('analyze.html', images=images, selected=name)
    return render_template('analyze.html', images=images, selected=None)

@app.route('/regions')
def regions():
    # results_cache contient les données de la dernière analyse faite dans /analyze
    return render_template('regions.html', 
                           selected=results_cache["selected"], 
                           regions=results_cache["regions"])

@app.route('/labeling', methods=['GET', 'POST'])
def labeling():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('labeling.html', images=images)

@app.route('/verrous')
def verrous():
    return render_template('verrous.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

if __name__ == '__main__':
    app.run(port=5001, debug=True)
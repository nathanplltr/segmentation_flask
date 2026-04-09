import os, json
from flask import Flask, render_template, request
from utils.segmentation_logic import process_image_segmentation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'

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
        name, regions = process_image_segmentation(path, app.config['PROCESSED_FOLDER'])
        results_cache["selected"] = name
        results_cache["regions"] = regions
        return render_template('analyze.html', images=images, selected=name)
    return render_template('analyze.html', images=images, selected=None)

@app.route('/regions')
def regions():
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

@app.route('/save_label', methods=['POST'])
def save_label():
    data = request.get_json()
    labels = data.get('labels', [])
    save_path = os.path.join('static', 'labels.json')
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    else:
        existing = []
    images_dans_payload = {l['image'] for l in labels}
    existing = [l for l in existing if l['image'] not in images_dans_payload]
    existing.extend(labels)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return json.dumps({'ok': True, 'count': len(labels)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
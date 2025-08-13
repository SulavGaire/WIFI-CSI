from flask import Flask, render_template, abort
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from io import BytesIO
import base64

app = Flask(__name__)

# === Configuration ===
DATA_ROOT = 'dataset'
IMAGE_DIR = os.path.join(DATA_ROOT, 'images')
CSI_MAT_DIR = os.path.join(DATA_ROOT, 'csi_mat')
MANIFEST_PATH = os.path.join(DATA_ROOT, 'manifest.jsonl')


def load_csi_samples():
    if not os.path.exists(MANIFEST_PATH):
        return []
    with open(MANIFEST_PATH) as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def get_image_base64(sample_id):
    path = os.path.join(IMAGE_DIR, f"{sample_id}.jpg")
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def get_csi_heatmap_base64(sample_id):
    path = os.path.join(CSI_MAT_DIR, f"{sample_id}.mat")
    if not os.path.exists(path):
        return None

    try:
        mat = loadmat(path)
        csi_data = mat.get('csi_data')
        if csi_data is None:
            return None

        magnitude = np.abs(csi_data)

        fig, ax = plt.subplots(figsize=(10, 4))
        heatmap = ax.imshow(magnitude.T, aspect='auto', interpolation='nearest')
        plt.colorbar(heatmap, ax=ax)
        ax.set_title(f'CSI Spectrogram - {sample_id}')
        ax.set_xlabel('Packet Index')
        ax.set_ylabel('Subcarrier Index')
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        print(f"[ERROR] Heatmap failed: {e}")
        return None

@app.route('/')
def home():
    return '<h2>Welcome to CSI Visualizer</h2><p><a href="/visualize">View Samples</a></p>'

@app.route('/visualize')
def visualize_index():
    samples = load_csi_samples()
    return render_template('csi_index.html', samples=samples)


@app.route('/visualize/<sample_id>')
def visualize_sample(sample_id):
    image_data = get_image_base64(sample_id)
    heatmap_data = get_csi_heatmap_base64(sample_id)

    if not image_data and not heatmap_data:
        return abort(404, f"No data for sample {sample_id}")

    return render_template(
        'csi_view.html',
        sample_id=sample_id,
        image_data=image_data,
        heatmap_data=heatmap_data
    )


if __name__ == '__main__':
    app.run(debug=True)

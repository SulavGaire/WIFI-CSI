import os
import numpy as np
from flask import Flask, render_template, request
from scipy.io import loadmat
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# ==== CONFIG ====
DATA_ROOT = "dataset"
WIN_LEN = 128
HOP = 64
RSSI_MIN, RSSI_MAX = -90, -10

app = Flask(__name__)

# ==== 1. LOAD MAT ====
def load_csi_mat(mat_path):
    mat = loadmat(mat_path)
    csi_data = mat['csi_data']  # shape: [num_packets, S] complex
    timestamps = mat.get('timestamps', None)
    if timestamps is not None:
        timestamps = timestamps.flatten()
    metadata = mat.get('metadata', None)
    return csi_data, timestamps, metadata

# ==== 2. CLEAN ====
def clean_csi(csi, metadata=None):
    # For MAT, usually already clean. CSV cleaning would go here if used.
    return csi

# ==== 3. PHASE SANITIZATION ====
def sanitize_phase(csi_complex):
    mags = np.abs(csi_complex)
    phases = np.unwrap(np.angle(csi_complex), axis=1)
    k = np.arange(csi_complex.shape[1])
    # Remove linear slope per packet
    for i in range(phases.shape[0]):
        p = np.polyfit(k, phases[i], 1)
        phases[i] -= p[0] * k + p[1]
    sanitized = mags * np.exp(1j * phases)
    return sanitized

# ==== 4. WINDOWING ====
def window_csi(csi, win_len, hop):
    clips = []
    for start in range(0, csi.shape[0] - win_len + 1, hop):
        clip = csi[start:start + win_len]
        clips.append(clip)
    return np.array(clips)  # [num_clips, T, S]

# ==== 5. VISUALIZATION HELPERS ====
def fig_to_base64(fig):
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def plot_csi_magnitude(csi_clip, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(np.abs(csi_clip).T, aspect='auto', origin='lower')
    ax.set_title(title + " - Magnitude")
    ax.set_xlabel("Time (packet index)")
    ax.set_ylabel("Subcarrier index")
    fig.colorbar(im, ax=ax, label="Magnitude")
    return fig_to_base64(fig)

def plot_csi_phase(csi_clip, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(np.angle(csi_clip).T, aspect='auto', origin='lower', cmap='twilight')
    ax.set_title(title + " - Phase")
    ax.set_xlabel("Time (packet index)")
    ax.set_ylabel("Subcarrier index")
    fig.colorbar(im, ax=ax, label="Phase (radians)")
    return fig_to_base64(fig)

def plot_constellation(csi_clip, title, pkt_idx=0):
    pkt = csi_clip[pkt_idx]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(pkt.real, pkt.imag, alpha=0.6)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(f"{title} - Constellation (Packet {pkt_idx})")
    ax.grid(True)
    return fig_to_base64(fig)

# ==== 6. DATA LIST ====
def list_all_samples():
    """Return list of (tag, sample_name, path) for all MAT files."""
    samples = []
    for tag in os.listdir(DATA_ROOT):
        tag_dir = os.path.join(DATA_ROOT, tag, "csi_mat")
        if os.path.isdir(tag_dir):
            for fname in os.listdir(tag_dir):
                if fname.endswith(".mat"):
                    samples.append((tag, fname[:-4], os.path.join(tag_dir, fname)))
    return samples

# ==== 7. ROUTES ====
@app.route("/", methods=["GET", "POST"])
def visualization():
    samples = list_all_samples()
    if request.method == "POST":
        selected = request.form.getlist("samples")
        plots = []
        for sel in selected:
            tag, sample = sel.split("::")
            path = os.path.join(DATA_ROOT, tag, "csi_mat", f"{sample}.mat")
            csi_data, ts, meta = load_csi_mat(path)

            # Get S, A, L
            S = csi_data.shape[1]
            A = 1
            L = 1

            # Sanitize & window
            csi_sanitized = sanitize_phase(csi_data)
            clips = window_csi(csi_sanitized, WIN_LEN, HOP)
            if len(clips) == 0:
                continue
            clip0 = clips[0]  # first clip for visualization

            mag_img = plot_csi_magnitude(clip0, f"{tag}/{sample}")
            phase_img = plot_csi_phase(clip0, f"{tag}/{sample}")
            const_img = plot_constellation(clip0, f"{tag}/{sample}")

            plots.append({
                "title": f"{tag}/{sample} [T={WIN_LEN}, S={S}, A={A}, L={L}]",
                "mag_img": mag_img,
                "phase_img": phase_img,
                "const_img": const_img
            })

        return render_template("visualization.html", samples=samples, plots=plots)

    return render_template("visualization.html", samples=samples, plots=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

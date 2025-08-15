import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import csv
from collections import Counter

# ==== CONFIG ====
MAT_PATH = "/home/noha/Documents/WIFI-CSI/Data Collection/dataset/Mouse/csi_mat/20250812_065649_791.mat"  # your MAT file
CSV_PATH = "dataset/<TAG>/csi_csv/<SAMPLE_ID>.csv"  # optional CSV
WIN_LEN = 128  # T value
HOP = 64       # overlap hop
RSSI_MIN, RSSI_MAX = -90, -10

# ==== 1. LOAD MAT ====
def load_csi_mat(mat_path):
    mat = loadmat(mat_path)
    csi_data = mat['csi_data']  # shape: [num_packets, S] complex
    timestamps = mat['timestamps'].flatten()
    metadata = mat['metadata'][0,0] if 'metadata' in mat else None
    return csi_data, timestamps, metadata

# ==== 2. CLEAN ====
def clean_csi(csi, metadata=None):
    # For MAT, data is already complex. CSV cleaning is where we check metadata.
    # If you want to use CSV instead, uncomment and parse.
    return csi

# ==== 3. PHASE SANITIZATION ====
def sanitize_phase(csi_complex):
    mags = np.abs(csi_complex)
    phases = np.unwrap(np.angle(csi_complex), axis=1)
    k = np.arange(csi_complex.shape[1])
    # Remove linear slope per packet
    for i in range(phases.shape[0]):
        p = np.polyfit(k, phases[i], 1)
        phases[i] -= p[0]*k + p[1]
    sanitized = mags * np.exp(1j * phases)
    return sanitized

# ==== 4. WINDOWING ====
def window_csi(csi, win_len, hop):
    clips = []
    for start in range(0, csi.shape[0] - win_len + 1, hop):
        clip = csi[start:start+win_len]
        clips.append(clip)
    return np.array(clips)  # [num_clips, T, S]

# ==== 5. VISUALIZATION ====
def plot_csi_magnitude(csi_clip):
    plt.imshow(np.abs(csi_clip).T, aspect='auto', origin='lower')
    plt.colorbar(label="Magnitude")
    plt.xlabel("Time (packet index)")
    plt.ylabel("Subcarrier index")
    plt.title("CSI Magnitude Heatmap")
    plt.show()

def plot_csi_phase(csi_clip):
    plt.imshow(np.angle(csi_clip).T, aspect='auto', origin='lower', cmap='twilight')
    plt.colorbar(label="Phase (radians)")
    plt.xlabel("Time (packet index)")
    plt.ylabel("Subcarrier index")
    plt.title("CSI Phase Heatmap")
    plt.show()

def plot_constellation(csi_clip, pkt_idx=0):
    pkt = csi_clip[pkt_idx]
    plt.scatter(pkt.real, pkt.imag, alpha=0.6)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title(f"I/Q Constellation (Packet {pkt_idx})")
    plt.grid(True)
    plt.show()

# ==== RUN ====
csi_data, ts, meta = load_csi_mat(MAT_PATH)

# Get S, A, L
S = csi_data.shape[1]
A = 1
L = 1
print(f"Detected S={S}, A={A}, L={L}")

# Sanitize phase
csi_sanitized = sanitize_phase(csi_data)

# Window into [num_clips, T, S]
clips = window_csi(csi_sanitized, WIN_LEN, HOP)
print(f"Final tensor shape (no A,L dims): {clips.shape}")  # should be [num_clips, T, S]

# Expand to [num_clips, T, S, A, L]
clips_4d = clips[..., np.newaxis, np.newaxis]
print(f"Final 4D tensor shape: {clips_4d.shape} [T={WIN_LEN}, S={S}, A={A}, L={L}]")

# Visualization example
plot_csi_magnitude(clips[0])
plot_csi_phase(clips[0])
plot_constellation(clips[0])

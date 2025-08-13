import os
import csv
import json
import time
import serial
import threading
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from scipy.io import savemat
from datetime import datetime, timedelta
from collections import deque
import ssl
import socket
import bisect

# Configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust for your setup
BAUD_RATE = 921600
CSI_BUFFER_SIZE = 10000
DATA_ROOT = 'dataset'
SSL_DIR = 'ssl'
IMAGE_FPS = 30  # Target frame rate

# CSI buffers and locks
csi_buffer = deque(maxlen=CSI_BUFFER_SIZE)
csi_timestamps = deque(maxlen=CSI_BUFFER_SIZE)
csi_lock = threading.Lock()
csi_available = threading.Event()

# Recording variables
is_recording = False
recording_start_time = None
recording_csi_buffer = deque()
recording_lock = threading.Lock()

DATA_COLUMNS_NAMES = [
    "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp",
    "ant", "sig_len", "rx_state", "len", "first_word", "data"
]

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB for image uploads

# SSL setup
SSL_CERT = os.path.join(SSL_DIR, 'cert.pem')
SSL_KEY = os.path.join(SSL_DIR, 'key.pem')
ssl_context = None

if os.path.exists(SSL_CERT) and os.path.exists(SSL_KEY):
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(SSL_CERT, SSL_KEY)
    print("SSL certificates found. HTTPS enabled.")
else:
    print("SSL certificates not found. Generating self-signed certificates...")
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"localhost")])
        cert = x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
            critical=False
        ).sign(private_key, hashes.SHA256())

        with open(SSL_KEY, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))
        with open(SSL_CERT, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(SSL_CERT, SSL_KEY)
        print("Self-signed certificates generated.")
    except ImportError:
        print("Warning: cryptography module not found. HTTPS disabled.")
        print("Install with: pip install cryptography")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/tags')
def get_tags():
    try:
        tags = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
        return jsonify(tags)
    except Exception as e:
        print(f"Error fetching tags: {e}")
        return jsonify([]), 500

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, recording_start_time, recording_csi_buffer
    with recording_lock:
        if is_recording:
            return jsonify(success=False, error='Already recording'), 409
        is_recording = True
        recording_start_time = datetime.utcnow()
        recording_csi_buffer = deque()
    return jsonify(success=True)

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording, recording_start_time, recording_csi_buffer
    with recording_lock:
        if not is_recording:
            return jsonify(success=False, error='Not recording'), 409
        is_recording = False
        csi_data_to_save = list(recording_csi_buffer)
        recording_start_time_copy = recording_start_time
        recording_csi_buffer = deque()

    tag_name = request.form.get('tag_name')
    if not tag_name:
        return jsonify(success=False, error='Missing tag_name'), 400

    if not all(c.isalnum() or c == '_' for c in tag_name):
        return jsonify(success=False, error='Tag name must be alphanumeric or underscore'), 400

    if not csi_data_to_save:
        return jsonify(success=False, error='No CSI data recorded'), 503

    sample_id = recording_start_time_copy.strftime("%Y%m%d_%H%M%S_%f")[:-3]

    image_dir = os.path.join(DATA_ROOT, tag_name, 'images', sample_id)
    csi_csv_dir = os.path.join(DATA_ROOT, tag_name, 'csi_csv')
    csi_mat_dir = os.path.join(DATA_ROOT, tag_name, 'csi_mat')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(csi_csv_dir, exist_ok=True)
    os.makedirs(csi_mat_dir, exist_ok=True)

    # Process images
    image_files = request.files.getlist('images')
    metadata_json = request.form.get('metadata')
    if not metadata_json:
        return jsonify(success=False, error='Missing image metadata'), 400
    
    try:
        image_metadata = json.loads(metadata_json)
    except json.JSONDecodeError:
        return jsonify(success=False, error='Invalid image metadata'), 400

    if len(image_files) != len(image_metadata):
        return jsonify(success=False, error='Image count mismatch'), 400

    # Save images and prepare timestamps
    image_timestamps = []
    for file, meta in zip(image_files, image_metadata):
        filename = f"{meta['index']}.jpg"
        file.save(os.path.join(image_dir, filename))
        image_timestamps.append(meta['timestamp'])

    # Associate CSI packets with images
    csi_times = [ts.timestamp() for ts in [item[2] for item in csi_data_to_save]]
    image_mapping = []
    for img_idx, img_time in enumerate(image_timestamps):
        try:
            if img_time.endswith("Z"):
                img_time = img_time.replace("Z", "+00:00")
            img_ts = datetime.fromisoformat(img_time).timestamp()
            # Find closest CSI packet using binary search
            pos = bisect.bisect_left(csi_times, img_ts)
            if pos == 0:
                closest_idx = 0
            elif pos == len(csi_times):
                closest_idx = len(csi_times) - 1
            else:
                before = csi_times[pos-1]
                after = csi_times[pos]
                closest_idx = pos-1 if abs(img_ts - before) < abs(img_ts - after) else pos
            image_mapping.append({
                'image_index': img_idx,
                'csi_index': closest_idx,
                'image_timestamp': img_time,
                'csi_timestamp': csi_data_to_save[closest_idx][2].isoformat()
            })
        except Exception as e:
            print(f"Error processing image timestamp: {e}")

    # Save CSI data to CSV
    csv_path = os.path.join(csi_csv_dir, f'{sample_id}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(DATA_COLUMNS_NAMES)
        for raw, _, _ in csi_data_to_save:
            reader = csv.reader([raw])
            writer.writerow(next(reader))

    # Save CSI data to MAT
    mat_path = os.path.join(csi_mat_dir, f'{sample_id}.mat')
    csi_complex_arrays = [parsed['complex_array'] for _, parsed, _ in csi_data_to_save]
    timestamps = [ts.isoformat() for _, _, ts in csi_data_to_save]
    savemat(mat_path, {
        'csi_data': np.array(csi_complex_arrays),
        'timestamps': timestamps,
        'metadata': csi_data_to_save[0][1]['metadata'],
        'image_mapping': image_mapping
    })

    manifest_entry = {
        'sample_id': sample_id,
        'tag_name': tag_name,
        'timestamp': datetime.utcnow().isoformat(),
        'image_dir': os.path.join(tag_name, 'images', sample_id),
        'csi_csv_path': os.path.join(tag_name, 'csi_csv', f'{sample_id}.csv'),
        'csi_mat_path': os.path.join(tag_name, 'csi_mat', f'{sample_id}.mat'),
        'num_csi_packets': len(csi_data_to_save),
        'num_images': len(image_files),
        'start_time': recording_start_time_copy.isoformat(),
        'end_time': csi_data_to_save[-1][2].isoformat() if csi_data_to_save else recording_start_time_copy.isoformat(),
        'first_packet_metadata': csi_data_to_save[0][1]['metadata'],
        'image_mapping': image_mapping
    }
    manifest_path = os.path.join(DATA_ROOT, 'manifest.jsonl')
    with open(manifest_path, 'a') as f:
        f.write(json.dumps(manifest_entry) + '\n')

    return jsonify(success=True, sample_id=sample_id, 
                   num_csi_packets=len(csi_data_to_save),
                   num_images=len(image_files))

@app.route('/csi-status')
def csi_status():
    with csi_lock:
        packet_count = len(csi_buffer)
        available = packet_count > 0
    return jsonify(available=available, packet_count=packet_count)

def parse_csi_packet(line):
    try:
        reader = csv.reader([line])
        parts = next(reader)
        if len(parts) != len(DATA_COLUMNS_NAMES):
            return None
        metadata = dict(zip(DATA_COLUMNS_NAMES[:-1], parts[:-1]))
        data_str = parts[-1].strip()
        if data_str.startswith('"[') and data_str.endswith(']"'):
            data_str = data_str[2:-2]
        elif data_str.startswith('[') and data_str.endswith(']'):
            data_str = data_str[1:-1]
        data_list = [int(x) for x in data_str.split(',')]
        data_len = int(metadata['len'])
        complex_array = np.zeros(data_len // 2, dtype=np.complex64)
        for i in range(data_len // 2):
            complex_array[i] = complex(data_list[i * 2 + 1], data_list[i * 2])
        parsed = {'metadata': metadata, 'raw_data': data_list, 'complex_array': complex_array}
        return line, parsed
    except Exception as e:
        print(f"Error parsing CSI packet: {e}")
        return None

def csi_reader_thread():
    print(f"Starting CSI reader on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Failed to open serial port: {e}")
        return
    print("CSI reader started. Waiting for data...")
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line or not line.startswith("CSI_DATA"):
                continue
            result = parse_csi_packet(line)
            if result:
                raw_csi, parsed_csi = result
                ts = datetime.utcnow()
                with csi_lock:
                    csi_buffer.append((raw_csi, parsed_csi))
                    csi_timestamps.append(ts)
                if is_recording:
                    with recording_lock:
                        recording_csi_buffer.append((raw_csi, parsed_csi, ts))
                csi_available.set()
        except Exception as e:
            print(f"Error in CSI reader: {e}")
            time.sleep(1)

def start_csi_reader():
    thread = threading.Thread(target=csi_reader_thread, daemon=True)
    thread.start()
    return thread

if __name__ == '__main__':
    start_csi_reader()
    print("Starting web server...")
    host = '0.0.0.0'
    port = 5000
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"
    local_ip = get_local_ip()
    if ssl_context:
        print(f"Access interface at https://{local_ip}:{port}")
        app.run(host=host, port=port, ssl_context=ssl_context, threaded=True)
    else:
        print("Running without HTTPS. Mobile camera may not work.")
        print(f"Access interface at http://{local_ip}:{port}")
        app.run(host=host, port=port, threaded=True)
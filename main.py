import os
import time
import cv2
import torch
import torch.nn as nn
import numpy
from torchvision import transforms, models
from PIL import Image
import joblib
import pandas as pd
import warnings
import threading
from datetime import timedelta
from flask import Flask, render_template, jsonify
import random
from pathlib import Path 

warnings.filterwarnings("ignore", category=UserWarning)

# ================= CONFIG =================

PROJECT_ROOT = Path(__file__).resolve().parent
PROTOTYPE_DIR = PROJECT_ROOT / "models" 
PROTOTYPE_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = PROJECT_ROOT / "CSV" / "plankton_predictions_log.csv"
CLEAN_CSV = PROJECT_ROOT / "CSV"/ "plankton_unique_log.csv"
VIDEO_PATH = PROJECT_ROOT / "microscopic_video.mp4"

# Flask in-memory storage
cleaned_results = []

# Templates folder path
TEMPLATES_DIR = PROTOTYPE_DIR/ "templates"

# ================= CNN MODEL =================
class HierarchicalMobileNetV3(nn.Module):
    def __init__(self, num_orders, num_families, num_species):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT')
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Identity()
        self.order_head = nn.Linear(in_features, num_orders)
        self.family_head = nn.Linear(in_features, num_families)
        self.species_head = nn.Linear(in_features, num_species)

    def forward(self, x):
        features = self.backbone(x)
        out_order = self.order_head(features)
        out_family = self.family_head(features)
        out_species = self.species_head(features)
        return out_order, out_family, out_species

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load label encoders and models
le_order = joblib.load(os.path.join(PROTOTYPE_DIR, "label_encoder_order_resaved.pkl"))
le_family = joblib.load(os.path.join(PROTOTYPE_DIR, "label_encoder_family_resaved.pkl"))
le_species = joblib.load(os.path.join(PROTOTYPE_DIR, "label_encoder_species_resaved.pkl"))

num_order_classes = len(le_order.classes_)
num_family_classes = len(le_family.classes_)
num_species_classes = len(le_species.classes_)

cnn_model = HierarchicalMobileNetV3(
    num_orders=num_order_classes,
    num_families=num_family_classes,
    num_species=num_species_classes
)
cnn_model.load_state_dict(torch.load(os.path.join(PROTOTYPE_DIR, "best_plankton_model_25.pth"), map_location=device))
cnn_model.to(device)
cnn_model.eval()

#rf_model = joblib.load(os.path.join(PROTOTYPE_DIR, "cytometry_correction_model.joblib"))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================= BIOMASS FACTORS =================
conversion_factors = {
    'Bivalvia_Mollusca': 0.5,
    'calypptopsis_Euphausiacea': 2.5,
    'Chaetognatha': 1.8,
    'cyphonaute': 0.2,
    'Eumalacostraca': 3.0,
    'Foraminifera': 0.1,
    'Noctiluca_Noctilucaceae': 0.05,
    'Ostracoda': 0.3,
    'Annelida': 0.6,
    'Actiniaria': 1.2,
    'Branchiostoma': 1.5,
    'Crustacea': 2.0,
    'Ctenophora': 0.9,
    'Echinodermata': 2.8,
    'Appendicularia': 0.4,
    'Calanoida': 1.1,
    'Siphonophorae': 0.7,
    'Cirripedia': 1.3,
    'Cladocera': 0.8,
    'Cyclopoida': 0.9,
    'Harpacticoida': 0.4,
    'Isopoda': 1.7,
    'Obelia': 0.15,
    'Liriope tetraphylla': 0.25,
    'Atlanta': 0.35,
    'Monstrilloida': 0.2,
    'Mysida': 1.9,
    'Solmundella bitentaculata': 0.3,
    'Tomopteridae': 1.4,
    'cypris': 0.45,
    'Corycaeidae': 0.6,
    'Lubbockia': 0.55
}

def add_biomass_info(df):
    # Normalize species keys
    norm_factors = {k.lower(): v for k, v in conversion_factors.items()}
    df["species_norm"] = df["species"].str.lower().str.strip()
    df["conversion_factor_ugC_per_individual"] = df["species_norm"].map(norm_factors).fillna(0)
    df["biomass_ugC"] = df["count"] * df["conversion_factor_ugC_per_individual"]
    df["biomass_mgC"] = df["biomass_ugC"] / 1000
    return df.drop(columns=["species_norm"])


# ================= VIDEO + CNN THREAD =================
def video_classifier_thread(result_queue=None):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_id = 0
    columns = ["timestamp", "frame_id", "cnn_species"]

    if not os.path.exists(RAW_CSV):
        pd.DataFrame(columns=columns).to_csv(RAW_CSV, index=False)

    print(f"Processing video: {VIDEO_PATH}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        try:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                _, _, out_species = cnn_model(img_tensor)
                pred_species = torch.argmax(out_species, dim=1).item()
                species_label = le_species.inverse_transform([pred_species])[0]
            
            # fake_flow_features = [[
            #     random.randint(0, 3500),
            #     random.randint(50, 350000)
            # ]]
            #flow_cyto_species = rf_model.predict(fake_flow_features)[0]

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            row = {
                "timestamp": ts,
                "frame_id": frame_id,
                "cnn_species": species_label,
                #"flow_cyto_species": flow_cyto_species
            }

            pd.DataFrame([row]).to_csv(RAW_CSV, mode='a', header=False, index=False)

            if result_queue is not None:
                result_queue.put(row)

            #print(f"[{ts}] Frame {frame_id} | CNN: {species_label} | FlowCyto: {flow_cyto_species}")

        except Exception as e:
            print(f"Error processing frame {frame_id}: {e}")

    cap.release()
    print(" processing complete.")

# ================= CLEANUP THREAD =================
def cleanup_thread_threaded(raw_csv, clean_csv, cleaned_results, time_threshold_ms=100, poll_interval=5):
    time_threshold = timedelta(milliseconds=time_threshold_ms)
    last_len = 0

    while True:
        try:
            df = pd.read_csv(raw_csv)
            if df.empty:
                time.sleep(poll_interval)
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            unique_rows = []
            last_species = None
            last_time = None

            for _, row in df.iterrows():
                species = row["cnn_species"]
                ts = row["timestamp"]
                if last_species != species or (last_time and (ts - last_time) > time_threshold):
                    unique_rows.append(row)
                    last_species = species
                    last_time = ts

            sampled_rows = unique_rows[::15]

            clean_df = pd.DataFrame(sampled_rows)
            clean_df.to_csv(clean_csv, index=False)

            cleaned_results.clear()
            cleaned_results.extend(sampled_rows)

            if len(cleaned_results) != last_len:
                print(f"Cleaned log updated at {time.strftime('%H:%M:%S')}, {len(cleaned_results)} unique rows")
                last_len = len(cleaned_results)

        except Exception as e:
            print(f"Error while updating: {e}")

        time.sleep(poll_interval)

# ================= FLASK APP =================
app = Flask(__name__, template_folder=TEMPLATES_DIR)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/timeseries")
def timeseries():
    if not os.path.exists(CLEAN_CSV):
        return jsonify([])
    df = pd.read_csv(CLEAN_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df = df.rename(columns={"cnn_species": "species"})
    return jsonify(df.to_dict(orient="records"))

@app.route("/species_counts")
def species_counts():
    if not os.path.exists(CLEAN_CSV):
        return jsonify([])
    df = pd.read_csv(CLEAN_CSV)
    counts = df["cnn_species"].value_counts().reset_index()
    counts.columns = ["species", "count"]
    counts = add_biomass_info(counts)
    return jsonify(counts.to_dict(orient="records"))

# ================= MAIN =================
if __name__ == "__main__":
    import queue
    result_queue = queue.Queue()
    
    t1 = threading.Thread(target=video_classifier_thread, args=(result_queue,), daemon=True)
    t2 = threading.Thread(target=cleanup_thread_threaded, args=(RAW_CSV, CLEAN_CSV, cleaned_results), daemon=True)
    t1.start()
    t2.start()

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
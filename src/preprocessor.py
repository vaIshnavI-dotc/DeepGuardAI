import os
import librosa
import numpy as np
from tqdm import tqdm
from PIL import Image

# ==============================
# CONFIGURATION
# ==============================

BASE_PATH = "archive/LA"

PATHS = [
    {
        "proto": os.path.join(BASE_PATH, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"),
        "audio": os.path.join(BASE_PATH, "ASVspoof2019_LA_train/flac")
    },
    {
        "proto": os.path.join(BASE_PATH, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"),
        "audio": os.path.join(BASE_PATH, "ASVspoof2019_LA_dev/flac")
    }
]

LIMIT = 2000


real_count = 0
fake_count = 0

# Absolute dataset path (guaranteed correct)
PROJECT_ROOT = os.getcwd()
REAL_DIR = os.path.join(PROJECT_ROOT, "dataset", "real")
FAKE_DIR = os.path.join(PROJECT_ROOT, "dataset", "fake")



print("Saving to:", REAL_DIR)
print("Saving to:", FAKE_DIR)

# ==============================
# PROCESSING
# ==============================

for p_set in PATHS:
    with open(p_set["proto"], "r") as f:
        lines = f.readlines()

        for line in tqdm(lines):

            if real_count >= LIMIT and fake_count >= LIMIT:
                break

            parts = line.strip().split()
            if len(parts) < 5:
                continue

            file_id = parts[1]
            label = parts[4]

            audio_path = os.path.join(p_set["audio"], f"{file_id}.flac")
            if not os.path.exists(audio_path):
                continue

            # Load audio
            y, sr = librosa.load(audio_path, sr=16000, duration=3.0)

            # Mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)

            # Normalize to 0-255
            S_norm = 255 * (S_db - S_db.min()) / (S_db.max() - S_db.min())
            S_norm = S_norm.astype(np.uint8)

            img = Image.fromarray(S_norm)

            if label == "bonafide" and real_count < LIMIT:
                save_path = os.path.join(REAL_DIR, f"{file_id}.png")
                img.save(save_path)
                real_count += 1

            elif label == "spoof" and fake_count < LIMIT:
                save_path = os.path.join(FAKE_DIR, f"{file_id}.png")
                img.save(save_path)
                fake_count += 1

print("DONE")
print("Real:", real_count)
print("Fake:", fake_count)
import os
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import cv2
import matplotlib.cm as cm

st.set_page_config(page_title="DeepGuardAI", layout="wide")
st.title("🎙️ DeepGuardAI — Deepfake Voice Detector (ResNet18)")

MODEL_PATH = "deepfake_model.pth"

# --------- Load ResNet18 (matches your training) ----------
@st.cache_resource
def load_resnet18():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model '{MODEL_PATH}' not found in the same folder as app.py")
        st.stop()

    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)   # 2 classes

    state = torch.load(MODEL_PATH, map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    return m

model = load_resnet18()

# Same normalization as training
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --------- Audio -> Spectrogram Image ----------
def audio_to_spec_image(file_path, sr=16000, duration=3.0, n_mels=128):
    y, sr = librosa.load(file_path, sr=sr, duration=duration)

    # pad/cut
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    # normalize to 0..255
    S_norm = 255 * (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    S_norm = S_norm.astype(np.uint8)

    img = Image.fromarray(S_norm).convert("RGB")
    return img, y, sr, S_db

def generate_gradcam(model, input_tensor, target_class):
    gradients = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.layer4[-1]  # last block in layer4 for ResNet18

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    # Forward (NO torch.no_grad here!)
    output = model(input_tensor)
    model.zero_grad()

    # Backward for target class
    score = output[0, target_class]
    score.backward()

    # Get saved tensors
    acts = activations[0]         # [1, C, H, W]
    grads = gradients[0]          # [1, C, H, W]

    # Weights: global-average-pool the gradients
    weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = (weights * acts).sum(dim=1).squeeze(0)    # [H, W]

    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle_fwd.remove()
    handle_bwd.remove()

    return cam

# --------- Predict ----------
def predict(file_path):
    img, y, sr, S_db = audio_to_spec_image(file_path)
    x = img_transform(img).unsqueeze(0)  # [1,3,224,224]

    # ---- Forward for probabilities (no gradients needed)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

    # ---- Grad-CAM needs gradients, so call it OUTSIDE no_grad
    x_cam = x.clone().detach().requires_grad_(True)
    cam = generate_gradcam(model, x_cam, pred)

    # Class mapping assumption: {'fake':0,'real':1}
    if pred == 0:
        label = "AI-Generated Voice"
        fake_prob = conf
    else:
        label = "Real Voice"
        fake_prob = 1 - conf

    return label, conf, fake_prob, y, sr, S_db, cam

# --------- UI ----------
uploaded = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])

if uploaded:
    tmp = "temp_audio.wav"
    with open(tmp, "wb") as f:
        f.write(uploaded.getbuffer())

    label, conf, fake_prob, y, sr, S_db, cam = predict(tmp)

    # threat color
    if fake_prob > 0.7:
        color = "red"
        lvl = "CRITICAL"
    elif fake_prob > 0.4:
        color = "orange"
        lvl = "MEDIUM"
    else:
        color = "green"
        lvl = "LOW"

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Prediction")
        st.markdown(f"### <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        st.write(f"**Model Confidence:** {conf:.2f}")
        st.write(f"**Deepfake Probability:** {fake_prob:.2f}")
        st.write(f"**Threat Level:** **{lvl}**")

        st.subheader("Threat Meter")
        st.markdown(f"""
        <div style="background:#333;border-radius:18px;padding:6px;">
          <div style="width:{int(fake_prob*100)}%;background:{color};height:24px;border-radius:14px;"></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Waveform")
        fig, ax = plt.subplots()
        ax.plot(y)
        st.pyplot(fig)

        st.subheader("Spectrogram (dB)")
        fig2, ax2 = plt.subplots()
        librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax2)
        st.pyplot(fig2)
        st.subheader("Grad-CAM (Model Attention)")
        heatmap = cv2.resize(cam, (S_db.shape[1], S_db.shape[0]))
        heatmap = np.uint8(255 * heatmap)

        colored = cm.jet(heatmap)[:, :, :3]          # 0..1 RGB
        colored = (colored * 255).astype(np.uint8)   # 0..255 RGB

        # Make a grayscale base image from spectrogram for overlay
        base = (255 * (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)).astype(np.uint8)
        base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

        overlay = cv2.addWeighted(base_rgb, 0.6, colored, 0.4, 0)
        st.image(overlay, caption="Red = high attention", use_container_width=True)

    try:
        os.remove(tmp)
    except:
        pass
else:
    st.info("Upload an audio file to start.")
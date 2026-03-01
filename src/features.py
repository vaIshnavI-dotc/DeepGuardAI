import torch
import librosa
import numpy as np

def extract_features(file_path, sr=16000, duration=3.0, n_mels=128):
    """
    Returns:
      features: torch.FloatTensor of shape [1, 1, 32, 32] (works with your MyCNN)
      y: waveform (numpy)
      sr: sample rate
    """
    # Load audio
    y, sr = librosa.load(file_path, sr=sr, duration=duration)

    # If audio shorter than duration, pad it
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalize to 0..1
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

    # Resize/Downsample to 32x32 to match your MyCNN fc layer expectation
    # (Your MyCNN expects 32*6*6 after conv/pool — this size works reliably)
    S_small = librosa.util.fix_length(S_norm, size=32, axis=1)  # time axis to 32
    if S_small.shape[0] != 32:
        # force mel axis to 32
        S_small = np.resize(S_small, (32, 32))

    # Tensor shape: [1, 1, 32, 32]
    features = torch.tensor(S_small, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return features, y, sr
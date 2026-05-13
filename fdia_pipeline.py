"""
=============================================================================
FDIA PIPELINE — Airbus Helicopter Accelerometer Dataset
=============================================================================
Author      : Adhith (with MIT-peer assist)
Purpose     : Dual-use — Airbus FYI Round 2 deliverable +
              IEEE Transactions on Aerospace and Electronic Systems paper
Target HW   : Development on Colab/CPU; deployment target NXP MCX (later)

STRUCTURE
---------
Phase 1 : Data loading & EDA
Phase 2 : Preprocessing & windowing
Phase 3 : Baseline anomaly detector (CAE on scalogram — best from paper)
Phase 4 : Phase 1 evaluation — healthy vs. natural fault classification
Phase 5 : Adversarial attack injection (FDIA threat model)
Phase 6 : Phase 2 evaluation — can the detector catch injected attacks?
Phase 7 : Metrics, plots, results table (paper-ready)

PAPER NOTATION (used in comments throughout)
--------------------------------------------
x(t)        : raw accelerometer signal, sampled at fs = 1024 Hz
x̃(t)        : adversarially perturbed signal
δ(t)        : adversarial perturbation = x̃(t) - x(t)
τ           : detection threshold (99th percentile of training residuals)
Res_k       : reconstruction residual of k-th sub-window
ε           : attack budget (L∞ norm bound on δ)
=============================================================================
"""

# ============================================================
# 0. DEPENDENCIES
# ============================================================
# Run this in Colab if needed:
# !pip install h5py numpy scipy matplotlib scikit-learn torch torchvision tqdm

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sp_signal
from sklearn.metrics import (roc_auc_score, f1_score, roc_curve,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# 1. CONFIGURATION  —  edit these paths for your setup
# ============================================================

# ---- SET THESE ----
# Absolute path to the folder containing dftrain.h5, dfvalid.h5, dfvalid_groundtruth.csv
DATA_DIR  = os.path.expanduser(
    "~/Airbus_Fly_Your_Ideas_2026/Project_Spectra/data/airbus_heli"
)
TRAIN_H5  = os.path.join(DATA_DIR, "dftrain.h5")
TEST_H5   = os.path.join(DATA_DIR, "dfvalid.h5")
LABEL_CSV = os.path.join(DATA_DIR, "dfvalid_groundtruth.csv")  # seqID, anomaly
# -------------------

# Signal parameters (from paper + dataset spec)
FS          = 1024          # sampling frequency, Hz
SEQ_LEN     = 61_440        # samples per sequence = 60 s × 1024 Hz
WINDOW_LEN  = 512           # sub-window length (0.5 s)
N_WINDOWS   = 120           # SEQ_LEN // WINDOW_LEN

# Scalogram parameters (best performer in paper, AUC 92%)
N_SCALES    = 64            # wavelet scales → image height
IMG_SIZE    = 64            # output image: 64 × 64

# Training
BATCH_SIZE  = 64
EPOCHS      = 15
LR          = 1e-3
BOTTLENECK  = 300
THRESHOLD_Q = 99            # percentile for τ

# Attack parameters (Phase 5)
ATTACK_TYPES = ["gaussian_noise", "bias_injection", "scaling", "replay", "fgsm_like"]
EPSILON      = 0.05         # L∞ budget as fraction of signal std

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CONFIG] Device: {DEVICE}")


# ============================================================
# 2. DATA LOADING
# ============================================================

def load_train_h5(filepath: str):
    """
    Load training HDF5 file (Pandas format saved with df.to_hdf()).

    File structure (confirmed):
        /dftrain/block0_values  — shape (1677, 61440)  float64
        /dftrain/axis1          — shape (1677,)         row indices

    All training sequences are healthy → labels all zero.

    Returns
    -------
    signals : np.ndarray, shape (1677, 61440)  float32
    labels  : np.ndarray, shape (1677,)         int32, all 0
    """
    fname = os.path.basename(filepath)
    group = os.path.splitext(fname)[0]          # "dftrain"
    key   = f"{group}/block0_values"

    print(f"\n[H5] Loading training file: {fname}")
    with h5py.File(filepath, 'r') as f:
        signals = np.array(f[key], dtype=np.float32)

    labels = np.zeros(len(signals), dtype=np.int32)
    print(f"[H5] Training signals: {signals.shape}  (all healthy)")
    return signals, labels


def load_valid_h5(filepath: str, label_csv: str):
    """
    Load validation HDF5 file and match labels from CSV.

    File structure (confirmed):
        /dfvalid/block0_values  — shape (594, 61440)  float64
        /dfvalid/axis1          — shape (594,)         row indices 0..593

    CSV structure:
        seqID   — integer, maps to row index (0-based)
        anomaly — 0 = healthy, 1 = anomalous

    Returns
    -------
    signals : np.ndarray, shape (594, 61440)  float32
    labels  : np.ndarray, shape (594,)         int32
    """
    import csv

    fname = os.path.basename(filepath)
    group = os.path.splitext(fname)[0]          # "dfvalid"
    key   = f"{group}/block0_values"

    print(f"\n[H5] Loading validation file: {fname}")
    with h5py.File(filepath, 'r') as f:
        signals = np.array(f[key], dtype=np.float32)

    # Load labels from CSV — robust to varied column names
    print(f"[CSV] Loading labels from: {os.path.basename(label_csv)}")
    labels = np.zeros(len(signals), dtype=np.int32)

    with open(label_csv, 'r') as f:
        reader = csv.DictReader(f)
        # Normalise column names to lowercase
        for row in reader:
            row_lower = {k.lower().strip(): v for k, v in row.items()}
            # Find seqID column
            seq_id = None
            for col in ['seqid', 'seq_id', 'id', 'index']:
                if col in row_lower:
                    seq_id = int(row_lower[col])
                    break
            # Find anomaly column
            anomaly = None
            for col in ['anomaly', 'label', 'fault', 'y']:
                if col in row_lower:
                    anomaly = int(float(row_lower[col]))
                    break
            if seq_id is not None and anomaly is not None:
                if seq_id < len(labels):
                    labels[seq_id] = anomaly

    counts = np.bincount(labels)
    print(f"[CSV] Labels loaded — Healthy: {counts[0]}, Anomalous: {counts[1]}")
    print(f"[H5] Validation signals: {signals.shape}")
    return signals, labels


def load_h5_dataset(filepath: str, label_csv: str = None):
    """
    Unified loader. Auto-detects train vs validation by filename.
    Pass label_csv only for the validation file.
    """
    if 'train' in os.path.basename(filepath).lower():
        return load_train_h5(filepath)
    else:
        assert label_csv is not None, "label_csv required for validation file"
        return load_valid_h5(filepath, label_csv)


# ============================================================
# 3. PREPROCESSING & SCALOGRAM ENCODING
# ============================================================

def compute_global_stats(train_signals: np.ndarray):
    """
    Compute dataset-level mean and std for z-score normalization.
    Using global stats (not per-sample) preserves inter-sample relationships.
    This is the key modification from the paper — normalization must be
    computed on training set and applied identically to test + attacked signals.
    """
    mu  = train_signals.mean()
    std = train_signals.std()
    print(f"[PREPROC] Global μ={mu:.4f}, σ={std:.4f}")
    return mu, std


def normalize(signals: np.ndarray, mu: float, std: float) -> np.ndarray:
    return (signals - mu) / (std + 1e-8)


def scalogram_encode(window: np.ndarray, n_scales: int = N_SCALES,
                     img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Compute STFT spectrogram magnitude for one window.

    scipy.signal.cwt was removed in SciPy 1.12. We use STFT spectrogram
    which is the second-best performer in the paper (AUC 91%, F1 91%).
    Install PyWavelets (pip install PyWavelets) to restore true CWT.

    SP[f, t] = |STFT(x)[f, t]|  where window=Hann(126), stride=8
    Output: (64, 64) float32, normalized to [0,1] in log scale.
    """
    nperseg  = 126
    noverlap = 118  # stride = 8
    _, _, Zxx = sp_signal.stft(window, fs=FS, window='hann',
                               nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)   # shape: (64, T_out)

    # Trim/pad time axis to img_size
    if mag.shape[1] >= img_size:
        mag = mag[:, :img_size]
    else:
        mag = np.pad(mag, ((0, 0), (0, img_size - mag.shape[1])), mode='edge')

    # Trim freq axis to img_size
    mag = mag[:img_size, :]

    # Log scale for dynamic range
    mag = np.log1p(mag)

    # Normalize to [0, 1]
    mn, mx = mag.min(), mag.max()
    if mx > mn:
        mag = (mag - mn) / (mx - mn)

    return mag.astype(np.float32)


def encode_dataset(signals: np.ndarray, mu: float, std: float,
                   desc: str = "Encoding") -> np.ndarray:
    """
    Full pipeline: normalize → window → scalogram encode.

    Returns
    -------
    images : np.ndarray, shape (N * N_WINDOWS, 1, IMG_SIZE, IMG_SIZE)
    """
    norm_signals = normalize(signals, mu, std)
    all_images = []

    for sig in tqdm(norm_signals, desc=desc):
        for w in range(N_WINDOWS):
            window = sig[w * WINDOW_LEN : (w + 1) * WINDOW_LEN]
            img = scalogram_encode(window)
            all_images.append(img)

    images = np.array(all_images, dtype=np.float32)
    images = images[:, np.newaxis, :, :]    # add channel dim → (N, 1, 64, 64)
    print(f"[ENCODE] {desc}: output shape {images.shape}")
    return images


# ============================================================
# 4. CONVOLUTIONAL AUTOENCODER (CAE) — 2D
# ============================================================
# Architecture matches Table 1 of reference paper (2D-CAE variant).
# Encoder: two conv layers → flatten → bottleneck
# Decoder: mirror of encoder

class ConvEncoder(nn.Module):
    def __init__(self, bottleneck: int = BOTTLENECK):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=2),   # 64×32×32
            nn.LeakyReLU(0.3),
            nn.Conv2d(64, 128, kernel_size=2, stride=2), # 128×16×16
            nn.LeakyReLU(0.3),
        )
        self.flatten_dim = 128 * 16 * 16   # = 32768
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, bottleneck),
            nn.LeakyReLU(0.3),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ConvDecoder(nn.Module):
    def __init__(self, bottleneck: int = BOTTLENECK):
        super().__init__()
        self.flatten_dim = 128 * 16 * 16
        self.fc = nn.Sequential(
            nn.Linear(bottleneck, self.flatten_dim),
            nn.LeakyReLU(0.3),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),   # output in [0, 1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 16, 16)
        return self.deconv(x)


class CAE(nn.Module):
    def __init__(self, bottleneck: int = BOTTLENECK):
        super().__init__()
        self.encoder = ConvEncoder(bottleneck)
        self.decoder = ConvDecoder(bottleneck)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ============================================================
# 5. TRAINING
# ============================================================

def train_cae(train_images: np.ndarray) -> tuple:
    """
    Train CAE on healthy-only scalogram images.
    Loss: MSE between input and reconstruction (standard for anomaly detection CAE).
    Returns trained model and per-epoch loss history.
    """
    dataset = TensorDataset(torch.from_numpy(train_images))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, pin_memory=False)

    model = CAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    criterion = nn.MSELoss()

    loss_history = []
    print(f"\n[TRAIN] Starting CAE training for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            loss  = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        avg_loss = epoch_loss / len(dataset)
        loss_history.append(avg_loss)
        print(f"  Epoch [{epoch:02d}/{EPOCHS}]  Loss: {avg_loss:.6f}")

    return model, loss_history


# ============================================================
# 6. RESIDUAL COMPUTATION & THRESHOLD CALIBRATION
# ============================================================

def compute_residuals(model: CAE, images: np.ndarray,
                      batch_size: int = 256) -> np.ndarray:
    """
    Compute L1 reconstruction residual per image.

    Res_k = Σ_i |x^i_or - x^i_rc|    (Equation 1 from reference paper)

    Returns array of shape (N_images,).
    """
    model.eval()
    residuals = []
    dataset = TensorDataset(torch.from_numpy(images))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            # L1 residual per image
            res = (batch - recon).abs().sum(dim=(1, 2, 3))
            residuals.append(res.cpu().numpy())

    return np.concatenate(residuals)


def residuals_to_sequence_scores(residuals: np.ndarray,
                                 n_windows: int = N_WINDOWS) -> np.ndarray:
    """
    Aggregate per-window residuals to per-sequence anomaly score.
    Strategy: max residual over all windows of a sequence.
    (Matches paper: "we monitor the maximum residual over its sub-series")

    Returns array of shape (N_sequences,).
    """
    n_seqs = len(residuals) // n_windows
    residuals = residuals[:n_seqs * n_windows].reshape(n_seqs, n_windows)
    return residuals.max(axis=1)


def calibrate_threshold(train_residuals: np.ndarray,
                        percentile: int = THRESHOLD_Q) -> float:
    """
    Set detection threshold τ as the Q-th percentile of training residuals.
    Using Q=99 matches the paper. In practice, calibrate on a held-out
    healthy validation split to avoid threshold leakage.
    """
    tau = np.percentile(train_residuals, percentile)
    print(f"[THRESHOLD] τ = {tau:.4f}  (Q={percentile} of training residuals)")
    return tau


# ============================================================
# 7. PHASE 1 EVALUATION — Healthy vs. Natural Faults
# ============================================================

def evaluate_detector(scores: np.ndarray, labels: np.ndarray,
                      tau: float, tag: str = "Phase 1"):
    """
    Compute TPR, FPR, F1, AUC given sequence-level anomaly scores and labels.
    Prints a paper-ready results table row.
    """
    preds = (scores > tau).astype(int)

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))

    tpr  = tp / (tp + fn + 1e-9)
    fpr  = fp / (fp + tn + 1e-9)
    f1   = f1_score(labels, preds, zero_division=0)
    auc  = roc_auc_score(labels, scores)

    print(f"\n{'='*50}")
    print(f"  [{tag}] RESULTS")
    print(f"{'='*50}")
    print(f"  TPR  : {tpr*100:.1f}%")
    print(f"  FPR  : {fpr*100:.1f}%")
    print(f"  F1   : {f1*100:.1f}%")
    print(f"  AUC  : {auc*100:.1f}%")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"{'='*50}")

    return {"tpr": tpr, "fpr": fpr, "f1": f1, "auc": auc,
            "preds": preds, "scores": scores, "labels": labels}


# ============================================================
# 8. ADVERSARIAL ATTACK INJECTION  —  FDIA THREAT MODEL
# ============================================================
"""
THREAT MODEL (paper Section II, formal definition):

    Attacker goal      : Evasion — cause healthy sensor readings to be
                         misclassified as healthy, hiding an injected fault.
                         (False-negative attack on the detector.)

    Attacker knowledge : Black-box / gray-box — attacker knows the sensor
                         modality and sampling rate but NOT the CAE weights.

    Attacker capability: Can inject additive signal δ(t) at the sensor bus
                         (e.g., compromised DAQ firmware or sensor spoofing).
                         Constraint: ||δ||_∞ ≤ ε·σ_x
                         where σ_x = std of the healthy training distribution.

    Five attack types implemented below cover the standard FDIA taxonomy:
    1. Gaussian noise injection         — stochastic, broadband
    2. Bias / DC offset injection       — persistent scalar shift
    3. Scaling attack                   — multiplicative gain perturbation
    4. Replay attack                    — substitute a past healthy window
    5. FGSM-like gradient attack        — worst-case additive perturbation
       (approximated without model access using signal gradient proxy)
"""

def attack_gaussian_noise(signal: np.ndarray, epsilon: float,
                          sigma: float) -> np.ndarray:
    """
    δ(t) ~ N(0, (ε·σ)²)
    Broadband stochastic attack. Models sensor quantization noise injection.
    """
    noise = np.random.normal(0, epsilon * sigma, size=signal.shape)
    return signal + noise


def attack_bias_injection(signal: np.ndarray, epsilon: float,
                          sigma: float) -> np.ndarray:
    """
    δ(t) = ε·σ·sign(U)  where U ~ Uniform(-1,1)
    Constant DC offset. Models compromised sensor zero-point calibration.
    """
    bias = epsilon * sigma * np.sign(np.random.uniform(-1, 1))
    return signal + bias


def attack_scaling(signal: np.ndarray, epsilon: float,
                   sigma: float) -> np.ndarray:
    """
    x̃(t) = (1 + ε·α) · x(t),  α ~ Uniform(-1, 1)
    Multiplicative gain attack. Models compromised amplifier/ADC scaling.
    Note: this violates the L∞ additive model for large signals — we clip.
    """
    alpha = np.random.uniform(-1, 1)
    scale = 1.0 + epsilon * alpha
    perturbed = signal * scale
    # Clip so ||δ||_∞ ≤ ε·σ
    delta = perturbed - signal
    delta = np.clip(delta, -epsilon * sigma, epsilon * sigma)
    return signal + delta


def attack_replay(signal: np.ndarray, replay_bank: np.ndarray) -> np.ndarray:
    """
    Replace signal with a randomly selected healthy signal from replay bank.
    Models replay attack: adversary records healthy data and re-injects it
    to mask an actual fault condition.
    """
    idx = np.random.randint(0, len(replay_bank))
    return replay_bank[idx].copy()


def attack_fgsm_like(signal: np.ndarray, epsilon: float,
                     sigma: float) -> np.ndarray:
    """
    FGSM-inspired perturbation without model access.
    Approximation: perturb in the direction that maximally changes the
    signal's local spectral energy (proxy for reconstruction difficulty).

    δ(t) = ε·σ·sign(∇_x ||STFT(x)||²)
    Gradient approximated via finite difference on spectral energy.

    In the paper this will be described as a gray-box spectral evasion attack.
    """
    # Compute spectral energy gradient proxy via finite differences
    eps_fd = 1e-4
    f, _, Zxx = sp_signal.stft(signal, fs=FS, nperseg=64)
    energy_orig = np.abs(Zxx).sum()

    grad = np.zeros_like(signal)
    # Approximate gradient at 32 random time points (tractable on CPU)
    indices = np.random.choice(len(signal), size=min(32, len(signal)), replace=False)
    for i in indices:
        sig_plus = signal.copy()
        sig_plus[i] += eps_fd
        _, _, Zxx_plus = sp_signal.stft(sig_plus, fs=FS, nperseg=64)
        energy_plus = np.abs(Zxx_plus).sum()
        grad[i] = (energy_plus - energy_orig) / eps_fd

    delta = epsilon * sigma * np.sign(grad)
    return signal + delta


def inject_attacks(healthy_signals: np.ndarray, train_signals: np.ndarray,
                   attack_type: str, epsilon: float = EPSILON) -> np.ndarray:
    """
    Apply a specified attack to all signals in healthy_signals.

    Parameters
    ----------
    healthy_signals : (N, SEQ_LEN) — normalized healthy test signals
    train_signals   : (N, SEQ_LEN) — training signals (used as replay bank)
    attack_type     : one of ATTACK_TYPES
    epsilon         : L∞ budget as fraction of training std

    Returns
    -------
    attacked_signals : (N, SEQ_LEN) — perturbed signals
    """
    sigma = train_signals.std()
    attacked = np.zeros_like(healthy_signals)

    for i, sig in enumerate(healthy_signals):
        if attack_type == "gaussian_noise":
            attacked[i] = attack_gaussian_noise(sig, epsilon, sigma)
        elif attack_type == "bias_injection":
            attacked[i] = attack_bias_injection(sig, epsilon, sigma)
        elif attack_type == "scaling":
            attacked[i] = attack_scaling(sig, epsilon, sigma)
        elif attack_type == "replay":
            attacked[i] = attack_replay(sig, train_signals)
        elif attack_type == "fgsm_like":
            attacked[i] = attack_fgsm_like(sig, epsilon, sigma)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

    return attacked


# ============================================================
# 9. VISUALIZATION
# ============================================================

def plot_training_loss(loss_history: list):
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, marker='o', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("CAE Training Loss (Healthy Data Only)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    plt.show()
    print("[PLOT] Saved: training_loss.png")


def plot_roc_curves(results_dict: dict):
    """
    Overlay ROC curves for Phase 1 (natural faults) and all attack types.
    """
    plt.figure(figsize=(8, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for (tag, res), color in zip(results_dict.items(), colors):
        fpr_arr, tpr_arr, _ = roc_curve(res['labels'], res['scores'])
        auc = res['auc']
        plt.plot(fpr_arr, tpr_arr, label=f"{tag} (AUC={auc*100:.1f}%)",
                 linewidth=2, color=color)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=50%)')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — FDIA Detection Performance")
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    plt.show()
    print("[PLOT] Saved: roc_curves.png")


def plot_residual_distributions(residual_dict: dict, tau: float):
    """
    Histogram of anomaly scores for healthy, natural faults, and each attack.
    Vertical dashed line = threshold τ.
    """
    n = len(residual_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (tag, (scores, labels)) in zip(axes, residual_dict.items()):
        healthy_s  = scores[labels == 0]
        anomaly_s  = scores[labels == 1]
        bins = np.linspace(scores.min(), np.percentile(scores, 99.5), 50)
        ax.hist(healthy_s,  bins=bins, alpha=0.6, label='Healthy',   color='steelblue')
        ax.hist(anomaly_s,  bins=bins, alpha=0.6, label='Anomalous', color='tomato')
        ax.axvline(tau, color='black', linestyle='--', label=f'τ={tau:.1f}')
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("Anomaly Score (max residual)")
        ax.legend(fontsize=8)

    plt.suptitle("Residual Score Distributions", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("residual_distributions.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[PLOT] Saved: residual_distributions.png")


def plot_attacked_signal(original: np.ndarray, attacked_dict: dict,
                         mu: float, std: float, window: int = 0):
    """
    Visual comparison of original vs. attacked signal for one window.
    Shows both time domain and scalogram side by side.
    """
    t = np.arange(WINDOW_LEN) / FS
    n_attacks = len(attacked_dict)
    fig = plt.figure(figsize=(5 * (n_attacks + 1), 6))
    gs  = gridspec.GridSpec(2, n_attacks + 1)

    orig_window = original[window * WINDOW_LEN : (window + 1) * WINDOW_LEN]
    orig_sc     = scalogram_encode(normalize(orig_window[np.newaxis], mu, std)[0])

    # Original
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, orig_window, linewidth=0.8)
    ax.set_title("Original")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(orig_sc, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_title("Scalogram")
    ax2.set_xlabel("Time bins")
    ax2.set_ylabel("Scale")

    for col, (atk_name, atk_signal) in enumerate(attacked_dict.items(), start=1):
        atk_window = atk_signal[window * WINDOW_LEN : (window + 1) * WINDOW_LEN]
        atk_sc     = scalogram_encode(normalize(atk_window[np.newaxis], mu, std)[0])

        ax = fig.add_subplot(gs[0, col])
        ax.plot(t, atk_window, linewidth=0.8, color='tomato')
        ax.set_title(f"Attack: {atk_name}")
        ax.set_xlabel("Time (s)")

        ax2 = fig.add_subplot(gs[1, col])
        ax2.imshow(atk_sc, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title("Scalogram")
        ax2.set_xlabel("Time bins")

    plt.suptitle("Original vs. Attacked Signals (time domain + scalogram)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("signal_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[PLOT] Saved: signal_comparison.png")


def print_results_table(all_results: dict):
    """
    Print a LaTeX-ready results table for the paper.
    """
    print("\n" + "="*65)
    print(f"  {'Condition':<25} {'TPR%':>6} {'FPR%':>6} {'F1%':>6} {'AUC%':>6}")
    print("="*65)
    for tag, res in all_results.items():
        print(f"  {tag:<25} {res['tpr']*100:>5.1f}  {res['fpr']*100:>5.1f}  "
              f"{res['f1']*100:>5.1f}  {res['auc']*100:>5.1f}")
    print("="*65)

    print("\n[LaTeX Table Row Format]")
    for tag, res in all_results.items():
        print(f"  {tag} & {res['tpr']*100:.1f} & {res['fpr']*100:.1f} & "
              f"{res['f1']*100:.1f} & {res['auc']*100:.1f} \\\\")


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def main():
    print("\n" + "="*60)
    print("  FDIA PIPELINE — Airbus Helicopter Dataset")
    print("="*60)

    # ----------------------------------------------------------
    # PHASE 1A: Load data
    # ----------------------------------------------------------
    print("\n[PHASE 1A] Loading dataset...")
    train_signals, train_labels = load_h5_dataset(TRAIN_H5)
    test_signals,  test_labels  = load_h5_dataset(TEST_H5, label_csv=LABEL_CSV)

    # Sanity checks
    assert train_signals.shape[1] == SEQ_LEN, \
        f"Expected SEQ_LEN={SEQ_LEN}, got {train_signals.shape[1]}"
    assert set(np.unique(train_labels)).issubset({0}), \
        "Training set should be healthy-only (label=0)"
    assert set(np.unique(test_labels)).issubset({0, 1}), \
        "Test set should have labels 0 and 1"

    # ----------------------------------------------------------
    # PHASE 1B: Preprocessing
    # ----------------------------------------------------------
    print("\n[PHASE 1B] Computing global normalization statistics...")
    mu, std = compute_global_stats(train_signals)

    # ----------------------------------------------------------
    # PHASE 1C: Scalogram encoding
    # ----------------------------------------------------------
    print("\n[PHASE 1C] Encoding training set to scalograms...")
    train_images = encode_dataset(train_signals, mu, std, desc="Train encode")

    print("\n[PHASE 1C] Encoding test set to scalograms...")
    test_images  = encode_dataset(test_signals,  mu, std, desc="Test encode")

    # ----------------------------------------------------------
    # PHASE 1D: Train CAE on healthy data
    # ----------------------------------------------------------
    print("\n[PHASE 1D] Training CAE...")
    model, loss_history = train_cae(train_images)
    plot_training_loss(loss_history)

    # ----------------------------------------------------------
    # PHASE 1E: Calibrate threshold on training residuals
    # ----------------------------------------------------------
    print("\n[PHASE 1E] Calibrating detection threshold τ...")
    train_residuals = compute_residuals(model, train_images)
    train_scores    = residuals_to_sequence_scores(train_residuals)
    tau             = calibrate_threshold(train_residuals)

    # ----------------------------------------------------------
    # PHASE 1F: Evaluate on test set (natural faults)
    # ----------------------------------------------------------
    print("\n[PHASE 1F] Phase 1 evaluation — natural fault detection...")
    test_residuals = compute_residuals(model, test_images)
    test_scores    = residuals_to_sequence_scores(test_residuals)
    p1_results     = evaluate_detector(test_scores, test_labels, tau,
                                       tag="Phase 1: Natural Faults")

    # Collect all results for final table
    all_results = {"Natural Faults": p1_results}

    # ----------------------------------------------------------
    # PHASE 2: FDIA Attack Injection
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("  PHASE 2 — FDIA ADVERSARIAL ATTACK INJECTION")
    print("="*60)

    # Use only healthy test signals as the attack substrate
    # (we inject attacks INTO healthy signals to test evasion)
    healthy_test_signals = test_signals[test_labels == 0]
    print(f"[PHASE 2] Using {len(healthy_test_signals)} healthy test sequences as attack base")

    # Labels for attacked sequences:
    # Ground truth = 1 (anomalous), because an injected attack IS an anomaly
    attack_labels = np.ones(len(healthy_test_signals), dtype=np.int32)

    # For each attack type: inject → encode → compute residuals → evaluate
    attacked_signals_dict = {}
    residual_plot_dict    = {}

    for attack_type in ATTACK_TYPES:
        print(f"\n[PHASE 2] Injecting attack: {attack_type} (ε={EPSILON})")

        # Inject on raw (un-normalized) signals — attack happens at sensor level
        attacked_raw = inject_attacks(
            healthy_test_signals,
            train_signals,
            attack_type,
            epsilon=EPSILON
        )
        attacked_signals_dict[attack_type] = attacked_raw

        # Encode attacked signals
        atk_images   = encode_dataset(attacked_raw, mu, std,
                                      desc=f"  Encoding [{attack_type}]")

        # Compute residuals
        atk_residuals = compute_residuals(model, atk_images)
        atk_scores    = residuals_to_sequence_scores(atk_residuals)

        # For residual plots: combine healthy and attacked scores
        # healthy scores = scores of clean healthy test seqs
        healthy_test_images = encode_dataset(healthy_test_signals, mu, std,
                                             desc="  Encoding [healthy base]")
        healthy_res    = compute_residuals(model, healthy_test_images)
        healthy_scores = residuals_to_sequence_scores(healthy_res)

        combined_scores = np.concatenate([healthy_scores, atk_scores])
        combined_labels = np.concatenate([
            np.zeros(len(healthy_scores)),
            np.ones(len(atk_scores))
        ])
        residual_plot_dict[attack_type] = (combined_scores, combined_labels.astype(int))

        # Evaluate
        atk_results = evaluate_detector(
            combined_scores, combined_labels.astype(int), tau,
            tag=f"Phase 2: {attack_type}"
        )
        all_results[f"Attack: {attack_type}"] = atk_results

    # ----------------------------------------------------------
    # PHASE 3: Plots & Results Table
    # ----------------------------------------------------------
    print("\n[PHASE 3] Generating paper-ready plots...")

    plot_roc_curves(all_results)
    plot_residual_distributions(residual_plot_dict, tau)

    # Signal comparison plot (one example sequence)
    example_attacked = {
        atk: attacked_signals_dict[atk][0]
        for atk in ATTACK_TYPES
    }
    plot_attacked_signal(healthy_test_signals[0], example_attacked, mu, std)

    # Final results table
    print_results_table(all_results)

    # Save model
    torch.save(model.state_dict(), "cae_fdia.pt")
    print("\n[DONE] Model saved to cae_fdia.pt")
    print("[DONE] All outputs saved. Ready for paper write-up.")

    return model, all_results, tau


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    model, results, tau = main()
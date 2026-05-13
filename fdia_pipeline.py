"""
=============================================================================
PROJECT SPECTRA — FDIA PIPELINE v2
Signal Processing + Machine Learning Approach
=============================================================================
Author      : Adhith (with MIT-peer assist)
Purpose     : Airbus FYI Round 2 + IEEE Transactions on AES paper
Target HW   : Development on Colab/CPU; deployment target NXP MCX

ARCHITECTURE
------------
Phase 1 : Data loading + EDA
Phase 2 : Signal feature extraction (no images, pure signal processing)
Phase 3 : One-Class anomaly detector (Isolation Forest + OCSVM ensemble)
Phase 4 : Phase 1 evaluation — healthy vs natural fault detection
Phase 5 : FDIA adversarial attack injection (5 attack types)
Phase 6 : Phase 2 evaluation — detection of injected attacks
Phase 7 : Goodfellow adversarial training — harden the detector
Phase 8 : Phase 3 evaluation — post-hardening performance
Phase 9 : Paper-ready plots + results table

PAPER NOTATION
--------------
x(t)     : raw accelerometer signal, fs=1024 Hz
f(x)     : feature vector extracted from x (8-dimensional)
δ        : adversarial perturbation in signal space
ε        : L∞ attack budget (fraction of training std)
A(f)     : anomaly score from ensemble detector
τ        : detection threshold (calibrated on training set)
=============================================================================
"""

# ============================================================
# 0. DEPENDENCIES
# ============================================================
import os, csv, time, warnings
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sp_signal
from scipy.stats import kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ============================================================
# 1. CONFIGURATION
# ============================================================

# ---- SET THIS ----
DATA_DIR  = os.path.expanduser(
    "~/Airbus_Fly_Your_Ideas_2026/Project_Spectra/data/airbus_heli"
)
TRAIN_H5  = os.path.join(DATA_DIR, "dftrain.h5")
TEST_H5   = os.path.join(DATA_DIR, "dfvalid.h5")
LABEL_CSV = os.path.join(DATA_DIR, "dfvalid_groundtruth.csv")
# ------------------

# Signal parameters
FS          = 1024      # Hz
SEQ_LEN     = 61_440    # 60s × 1024 Hz
WINDOW_LEN  = 512       # 0.5s per window
N_WINDOWS   = SEQ_LEN // WINDOW_LEN   # = 120

# Attack parameters
EPSILON      = 0.10     # L∞ budget as fraction of signal std
ATTACK_TYPES = ["gaussian_noise", "bias_injection", "scaling",
                "replay", "fgsm_like"]

# Adversarial training
ADV_ALPHA   = 0.5       # mixture: 0.5 clean + 0.5 adversarial (Goodfellow 2015)
ADV_EPSILON = 0.10      # perturbation budget for adversarial training

print(f"[CONFIG] DATA_DIR: {DATA_DIR}")
print(f"[CONFIG] Window: {WINDOW_LEN} samples ({WINDOW_LEN/FS*1000:.0f}ms) | "
      f"Windows/seq: {N_WINDOWS}")


# ============================================================
# 2. DATA LOADING
# ============================================================

def load_train_h5(filepath):
    fname = os.path.basename(filepath)
    group = os.path.splitext(fname)[0]
    with h5py.File(filepath, 'r') as f:
        signals = np.array(f[f'{group}/block0_values'], dtype=np.float32)
    labels = np.zeros(len(signals), dtype=np.int32)
    print(f"[DATA] Train: {signals.shape}  (all healthy)")
    return signals, labels


def load_valid_h5(filepath, label_csv):
    fname = os.path.basename(filepath)
    group = os.path.splitext(fname)[0]
    with h5py.File(filepath, 'r') as f:
        signals = np.array(f[f'{group}/block0_values'], dtype=np.float32)
    labels = np.zeros(len(signals), dtype=np.int32)
    with open(label_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = {k.lower().strip(): v for k, v in row.items()}
            sid = int(float(r.get('seqid', r.get('id', 0))))
            lbl = int(float(r.get('anomaly', r.get('label', 0))))
            if sid < len(labels):
                labels[sid] = lbl
    counts = np.bincount(labels)
    print(f"[DATA] Valid: {signals.shape}  "
          f"Healthy={counts[0]}  Anomalous={counts[1]}")
    return signals, labels


# ============================================================
# 3. SIGNAL FEATURE EXTRACTION
# ============================================================
"""
Feature vector f(x) for one window x of length WINDOW_LEN:

  f1 : RMS energy       = sqrt(mean(x²))
  f2 : Peak-to-peak     = max(x) - min(x)
  f3 : Kurtosis         = E[(x-μ)⁴] / σ⁴  (sensitive to impulsive faults)
  f4 : Crest factor     = max(|x|) / RMS   (ratio of peak to RMS)
  f5 : Zero crossing rate = # sign changes / length
  f6 : Spectral entropy = -Σ p_i log(p_i)  where p_i = PSD_i / Σ PSD
  f7 : Dominant frequency = argmax(|FFT(x)|²)  in Hz
  f8 : Band energy ratio = energy(0-100Hz) / energy(total)
       (rotor harmonics concentrate below 100Hz in helicopters)

These 8 features are computed per window then aggregated to sequence level
by taking [mean, std, max] across all N_WINDOWS windows → 24-dim feature vector.

Why these features:
- RMS + peak-to-peak: directly capture the 10-30x amplitude anomaly we saw
- Kurtosis: classic bearing fault indicator, sensitive to impulses
- Crest factor: ratio-based, robust to operating condition changes
- Spectral entropy: healthy signals have structured spectra (low entropy);
  faults add broadband noise (high entropy)
- Dominant freq + band energy: rotor imbalance shifts spectral content
"""

def extract_window_features(window: np.ndarray) -> np.ndarray:
    """
    Extract 8 features from one window of WINDOW_LEN samples.
    Returns np.ndarray of shape (8,).
    """
    n   = len(window)
    eps = 1e-12

    # f1: RMS energy
    rms = np.sqrt(np.mean(window ** 2))

    # f2: Peak-to-peak amplitude
    p2p = window.max() - window.min()

    # f3: Kurtosis
    kurt = kurtosis(window, fisher=True)   # excess kurtosis (0 for Gaussian)

    # f4: Crest factor
    crest = np.max(np.abs(window)) / (rms + eps)

    # f5: Zero crossing rate
    zcr = np.sum(np.diff(np.sign(window)) != 0) / n

    # f6, f7, f8: Spectral features via FFT
    fft_mag  = np.abs(np.fft.rfft(window * np.hanning(n)))
    freqs    = np.fft.rfftfreq(n, d=1.0/FS)
    psd      = fft_mag ** 2
    psd_norm = psd / (psd.sum() + eps)

    # Spectral entropy
    spec_ent = -np.sum(psd_norm * np.log(psd_norm + eps))

    # Dominant frequency (Hz)
    dom_freq = freqs[np.argmax(psd)]

    # Band energy ratio: energy below 100 Hz / total
    low_band_mask = freqs <= 100.0
    band_ratio    = psd[low_band_mask].sum() / (psd.sum() + eps)

    return np.array([rms, p2p, kurt, crest, zcr,
                     spec_ent, dom_freq, band_ratio], dtype=np.float32)


def extract_sequence_features(signal: np.ndarray) -> np.ndarray:
    """
    Extract feature vector for one full sequence.

    For each of N_WINDOWS windows, compute 8 features.
    Aggregate across windows using [mean, std, max] → 24-dim vector.

    The aggregation captures:
    - mean: typical operating condition
    - std:  variability (faults increase variability)
    - max:  worst-case window (catches localized fault bursts)
    """
    window_feats = np.zeros((N_WINDOWS, 8), dtype=np.float32)
    for w in range(N_WINDOWS):
        window = signal[w * WINDOW_LEN : (w + 1) * WINDOW_LEN]
        window_feats[w] = extract_window_features(window)

    # Aggregate: mean, std, max across windows → (24,)
    feat_mean = window_feats.mean(axis=0)
    feat_std  = window_feats.std(axis=0)
    feat_max  = window_feats.max(axis=0)

    return np.concatenate([feat_mean, feat_std, feat_max])


def extract_dataset_features(signals: np.ndarray,
                              desc: str = "Extracting features") -> np.ndarray:
    """
    Extract 24-dim feature vectors for all sequences.
    Returns np.ndarray of shape (N_sequences, 24).
    """
    print(f"\n[FEATURES] {desc} for {len(signals)} sequences...")
    t0 = time.time()
    features = np.zeros((len(signals), 24), dtype=np.float32)
    for i, sig in enumerate(signals):
        features[i] = extract_sequence_features(sig)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(signals) - i - 1)
            print(f"  {i+1}/{len(signals)}  elapsed={elapsed:.0f}s  "
                  f"ETA={eta:.0f}s")
    print(f"  Done in {time.time()-t0:.1f}s  shape={features.shape}")
    return features


# ============================================================
# 4. FEATURE NAMES (for interpretability)
# ============================================================

FEATURE_NAMES = []
for stat in ['mean', 'std', 'max']:
    for feat in ['RMS', 'P2P', 'Kurtosis', 'Crest',
                 'ZCR', 'SpecEnt', 'DomFreq', 'BandRatio']:
        FEATURE_NAMES.append(f"{feat}_{stat}")


# ============================================================
# 5. ANOMALY DETECTOR — ENSEMBLE
# ============================================================
"""
Detector architecture: ensemble of two one-class classifiers.

1. Isolation Forest (Liu et al. 2008)
   - Builds random trees that isolate points
   - Anomalies are isolated in fewer splits (shorter path length)
   - Robust to high-dimensional data, no distributional assumption
   - contamination=0.01: expects ~1% of training data to be outliers

2. One-Class SVM (Schölkopf et al. 2001)
   - Learns a hypersphere in kernel space enclosing healthy data
   - RBF kernel captures nonlinear healthy manifold
   - nu=0.05: at most 5% of training samples are support vectors

Ensemble decision: average of normalized scores from both detectors.
This reduces variance and improves robustness over either alone.

Formal anomaly score:
    A(x) = 0.5 · A_IF(f(x)) + 0.5 · A_OCSVM(f(x))
where f(x) is the 24-dim feature vector.
Detection: flag as anomalous if A(x) > τ
"""

class FDIADetector:
    def __init__(self):
        self.iso_forest = IsolationForest(
            n_estimators=200,
            max_samples='auto',
            contamination=0.01,
            random_state=SEED,
            n_jobs=-1
        )
        self.ocsvm = OneClassSVM(
            kernel='rbf',
            nu=0.05,
            gamma='scale'
        )
        self.scaler = StandardScaler()
        self.tau    = None
        self.is_fit = False

    def fit(self, features: np.ndarray):
        """
        Train detector on healthy-only feature vectors.
        Scales features, fits both detectors.
        """
        print("\n[DETECTOR] Fitting on healthy training features...")
        X = self.scaler.fit_transform(features)

        print("  Fitting Isolation Forest...")
        self.iso_forest.fit(X)

        print("  Fitting One-Class SVM...")
        self.ocsvm.fit(X)

        self.is_fit = True
        print("[DETECTOR] Fitting complete.")

    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for feature vectors.
        Higher score = more anomalous.

        Isolation Forest: score_samples returns negative average path length
        (more negative = more anomalous). We negate to get positive anomaly score.

        OCSVM: decision_function returns signed distance from boundary
        (more negative = more anomalous). We negate.

        Both are then min-max normalized to [0,1] using training distribution,
        then averaged.
        """
        assert self.is_fit, "Detector not fitted yet"
        X = self.scaler.transform(features)

        # Raw scores (higher = more anomalous after negation)
        if_scores   = -self.iso_forest.score_samples(X)
        svm_scores  = -self.ocsvm.decision_function(X)

        # Normalize using training distribution stats stored at calibration
        if_norm  = (if_scores  - self._if_min)  / (self._if_range  + 1e-8)
        svm_norm = (svm_scores - self._svm_min) / (self._svm_range + 1e-8)

        return 0.5 * if_norm + 0.5 * svm_norm

    def calibrate_threshold(self, train_features: np.ndarray,
                            percentile: int = 95) -> float:
        """
        Set detection threshold τ at Q-th percentile of training anomaly scores.
        Also stores normalization stats for score().

        Using Q=95 (vs paper's Q=99) because ensemble scores have lower variance
        than single-model reconstruction residuals.
        """
        X = self.scaler.transform(train_features)

        if_scores  = -self.iso_forest.score_samples(X)
        svm_scores = -self.ocsvm.decision_function(X)

        # Store normalization stats
        self._if_min   = if_scores.min()
        self._if_range = if_scores.max() - if_scores.min()
        self._svm_min  = svm_scores.min()
        self._svm_range = svm_scores.max() - svm_scores.min()

        # Compute ensemble scores on training set
        train_scores = self.score(train_features)
        self.tau = np.percentile(train_scores, percentile)
        print(f"[DETECTOR] τ = {self.tau:.4f}  (Q={percentile})")
        return self.tau


# ============================================================
# 6. EVALUATION
# ============================================================

def evaluate(scores: np.ndarray, labels: np.ndarray,
             tau: float, tag: str = "") -> dict:
    """
    Compute TPR, FPR, F1, AUC. Print results row.
    """
    preds = (scores > tau).astype(int)

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))

    tpr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    f1  = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, scores)

    print(f"  {tag:<30} TPR={tpr*100:5.1f}%  FPR={fpr*100:5.1f}%  "
          f"F1={f1*100:5.1f}%  AUC={auc*100:5.1f}%")

    return dict(tpr=tpr, fpr=fpr, f1=f1, auc=auc,
                preds=preds, scores=scores, labels=labels,
                tp=tp, fp=fp, fn=fn, tn=tn)


# ============================================================
# 7. FDIA ATTACK INJECTION
# ============================================================
"""
THREAT MODEL (formal, for paper Section II)
-------------------------------------------
Attacker goal      : Evasion — inject signal δ into healthy sensor reading x
                     such that the compromised reading x̃ = x + δ is classified
                     as healthy (false negative), hiding a fault condition.

Attacker capability: Additive perturbation at sensor/DAQ level.
                     Constraint: ||δ||_∞ ≤ ε · σ_train
                     where σ_train = std of healthy training distribution.

Attacker knowledge :
  - Gaussian / Bias / Scaling : zero-knowledge (no model access)
  - Replay                    : zero-knowledge + historical data access
  - FGSM-like                 : gray-box (knows feature extraction process,
                                not detector weights)

Five attack types cover the standard ICS/avionics FDIA taxonomy:
  Type I   : Stochastic noise injection (Gaussian)
  Type II  : Persistent bias / DC offset injection
  Type III : Multiplicative gain manipulation
  Type IV  : Replay attack (historical healthy data substitution)
  Type V   : Gradient-based spectral evasion (FGSM-inspired)
"""

def attack_gaussian_noise(signal, epsilon, sigma):
    """Type I: δ(t) ~ N(0, (ε·σ)²)"""
    return signal + np.random.normal(0, epsilon * sigma, signal.shape)


def attack_bias_injection(signal, epsilon, sigma):
    """Type II: δ(t) = ε·σ·sign(U), U~Uniform(-1,1)"""
    return signal + epsilon * sigma * np.sign(np.random.uniform(-1, 1))


def attack_scaling(signal, epsilon, sigma):
    """Type III: x̃(t) = (1 + ε·α)·x(t), α~Uniform(-1,1), clipped to L∞ budget"""
    alpha = np.random.uniform(-1, 1)
    delta = np.clip(signal * epsilon * alpha,
                    -epsilon * sigma, epsilon * sigma)
    return signal + delta


def attack_replay(signal, replay_bank):
    """Type IV: substitute randomly selected healthy recording"""
    return replay_bank[np.random.randint(len(replay_bank))].copy()


def attack_fgsm_like(signal, epsilon, sigma):
    """
    Type V: Gray-box FGSM-inspired attack.

    Perturbs signal in direction that maximally changes the feature vector,
    approximating ∇_x ||f(x)||² via finite differences.

    δ(t) = ε·σ·sign(∂/∂x_t [Σ_i f_i(x)²])

    This targets the feature extractor directly without requiring access
    to the detector's decision boundary (gray-box assumption).
    """
    eps_fd    = sigma * 1e-3
    f0        = extract_sequence_features(signal)
    grad      = np.zeros_like(signal)
    # Sample 64 time points for tractable finite difference
    idx       = np.random.choice(len(signal), size=64, replace=False)
    for i in idx:
        sp        = signal.copy()
        sp[i]    += eps_fd
        fp        = extract_sequence_features(sp)
        grad[i]   = np.sum((fp - f0) ** 2) / eps_fd
    delta = epsilon * sigma * np.sign(grad)
    return signal + delta


def inject_attacks(healthy_signals, train_signals, attack_type,
                   epsilon=EPSILON):
    """Apply attack to all signals. Returns attacked array."""
    sigma   = train_signals.std()
    attacked = np.zeros_like(healthy_signals)
    print(f"  Injecting [{attack_type}]  ε={epsilon}  σ_train={sigma:.4f}")
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
    return attacked


# ============================================================
# 8. GOODFELLOW ADVERSARIAL TRAINING
# ============================================================
"""
Adversarial training (Goodfellow et al. 2015) adapted to one-class detection:

Original formulation (for classifiers):
    J̃(θ,x,y) = α·J(θ,x,y) + (1-α)·J(θ, x+ε·sign(∇_xJ), y)

Our adaptation for one-class feature-space detector:
    We augment the training feature set with adversarially perturbed features:
    F̃_train = α·F_healthy ∪ (1-α)·F_attacked

    where F_attacked are features extracted from signals perturbed by
    FGSM-like attack at budget ε_adv.

    The detector is then refitted on F̃_train. This forces the detector to
    learn a decision boundary that is robust to the worst-case perturbations
    an attacker can apply within budget ε_adv.

    α = 0.5 matches Goodfellow et al.'s best reported hyperparameter.

Why this works for anomaly detection:
    The Isolation Forest and OCSVM learn the boundary of the healthy manifold.
    Adversarial training expands this boundary in the directions most vulnerable
    to perturbation, making it harder for an attacker to push a perturbed signal
    outside the boundary without exceeding the L∞ budget.
"""

def generate_adversarial_features(train_signals: np.ndarray,
                                   train_features: np.ndarray,
                                   detector: FDIADetector,
                                   epsilon: float = ADV_EPSILON) -> np.ndarray:
    """
    Generate adversarially perturbed feature vectors for adversarial training.

    For each healthy training signal, apply FGSM-like perturbation and
    extract features from the perturbed signal.

    Returns adversarial feature matrix of same shape as train_features.
    """
    print(f"\n[ADV TRAIN] Generating adversarial features  ε={epsilon}...")
    sigma     = train_signals.std()
    adv_feats = np.zeros_like(train_features)

    for i, sig in enumerate(train_signals):
        attacked     = attack_fgsm_like(sig, epsilon, sigma)
        adv_feats[i] = extract_sequence_features(attacked)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(train_signals)}")

    print(f"  Adversarial features shape: {adv_feats.shape}")
    return adv_feats


def adversarial_train(train_signals: np.ndarray,
                      train_features: np.ndarray,
                      alpha: float = ADV_ALPHA,
                      epsilon: float = ADV_EPSILON) -> FDIADetector:
    """
    Retrain detector on mixture of clean and adversarial features.

    Mixed training set:
        F̃ = [F_clean (α fraction)] ∪ [F_adversarial (1-α fraction)]

    Returns new hardened detector.
    """
    print(f"\n[ADV TRAIN] Adversarial training  α={alpha}  ε={epsilon}")

    # Generate adversarial features
    adv_features = generate_adversarial_features(
        train_signals, train_features, None, epsilon
    )

    # Mix: α clean + (1-α) adversarial
    n_clean = int(alpha * len(train_features))
    n_adv   = len(train_features) - n_clean

    clean_idx = np.random.choice(len(train_features), n_clean, replace=False)
    adv_idx   = np.random.choice(len(adv_features),   n_adv,   replace=False)

    mixed_features = np.vstack([
        train_features[clean_idx],
        adv_features[adv_idx]
    ])

    print(f"[ADV TRAIN] Mixed set: {n_clean} clean + {n_adv} adversarial "
          f"= {len(mixed_features)} total")

    # Refit detector on mixed set
    hardened = FDIADetector()
    hardened.fit(mixed_features)
    hardened.calibrate_threshold(mixed_features, percentile=95)

    return hardened


# ============================================================
# 9. VISUALIZATION
# ============================================================

def plot_feature_distributions(train_feats, test_feats, test_labels,
                                n_features=8):
    """
    Plot distribution of each feature for healthy vs anomalous sequences.
    Shows which features are most discriminative.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    healthy_feats  = test_feats[test_labels == 0]
    anomaly_feats  = test_feats[test_labels == 1]
    feat_names_short = [n.replace('_mean','') for n in FEATURE_NAMES[:8]]

    for i in range(n_features):
        ax = axes[i]
        # Use mean aggregation features (first 8)
        h_vals = healthy_feats[:, i]
        a_vals = anomaly_feats[:, i]
        t_vals = train_feats[:, i]

        bins = np.linspace(
            min(h_vals.min(), a_vals.min()),
            min(max(h_vals.max(), a_vals.max()),
                np.percentile(a_vals, 99)),
            40
        )
        ax.hist(t_vals, bins=bins, alpha=0.4, label='Train (healthy)',
                color='gray', density=True)
        ax.hist(h_vals, bins=bins, alpha=0.6, label='Test healthy',
                color='steelblue', density=True)
        ax.hist(a_vals, bins=bins, alpha=0.6, label='Test anomalous',
                color='tomato', density=True)
        ax.set_title(feat_names_short[i], fontsize=10)
        ax.legend(fontsize=7)

    plt.suptitle('Feature Distributions: Healthy vs Anomalous', fontsize=13)
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[PLOT] Saved: feature_distributions.png")


def plot_roc_curves(results_dict: dict):
    plt.figure(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for (tag, res), color in zip(results_dict.items(), colors):
        fpr_arr, tpr_arr, _ = roc_curve(res['labels'], res['scores'])
        linestyle = '-' if 'Phase 3' in tag or 'Hardened' in tag else '--'
        plt.plot(fpr_arr, tpr_arr,
                 label=f"{tag}  (AUC={res['auc']*100:.1f}%)",
                 linewidth=2, color=color, linestyle=linestyle)

    plt.plot([0,1],[0,1],'k:',linewidth=1,label='Random (50%)')
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — FDIA Detection (solid=hardened, dashed=baseline)",
              fontsize=12)
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150)
    plt.show()
    print("[PLOT] Saved: roc_curves.png")


def plot_score_distributions(results_dict: dict, tau: float,
                              tau_hardened: float = None):
    n   = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (tag, res) in zip(axes, results_dict.items()):
        h = res['scores'][res['labels'] == 0]
        a = res['scores'][res['labels'] == 1]
        mn = min(h.min(), a.min())
        mx = max(np.percentile(h,99.5), np.percentile(a,99.5))
        bins = np.linspace(mn, mx, 50)
        ax.hist(h, bins=bins, alpha=0.6, label='Healthy',   color='steelblue',
                density=True)
        ax.hist(a, bins=bins, alpha=0.6, label='Anomalous', color='tomato',
                density=True)
        ax.axvline(tau, color='black', linestyle='--',
                   linewidth=1.5, label=f'τ={tau:.3f}')
        if tau_hardened:
            ax.axvline(tau_hardened, color='green', linestyle=':',
                       linewidth=1.5, label=f'τ_hard={tau_hardened:.3f}')
        ax.set_title(tag, fontsize=9)
        ax.set_xlabel("Anomaly Score")
        ax.legend(fontsize=7)

    plt.suptitle("Anomaly Score Distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('score_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[PLOT] Saved: score_distributions.png")


def plot_signal_examples(train_signals, test_signals, test_labels):
    """Quick sanity check: plot healthy vs anomalous raw signals."""
    h_idx = np.where(test_labels == 0)[0][:2]
    a_idx = np.where(test_labels == 1)[0][:2]
    t     = np.arange(2048) / FS

    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    for i, idx in enumerate(h_idx):
        axes[0,i].plot(t, test_signals[idx,:2048], linewidth=0.5,
                       color='steelblue')
        axes[0,i].set_title(f'Healthy seqID={idx}')
        axes[0,i].set_ylabel('Amplitude')
        axes[0,i].set_xlabel('Time (s)')

    for i, idx in enumerate(a_idx):
        axes[1,i].plot(t, test_signals[idx,:2048], linewidth=0.5,
                       color='tomato')
        axes[1,i].set_title(f'Anomalous seqID={idx}')
        axes[1,i].set_ylabel('Amplitude')
        axes[1,i].set_xlabel('Time (s)')

    plt.suptitle('Raw Signal: Healthy vs Anomalous', fontsize=13)
    plt.tight_layout()
    plt.savefig('raw_signals.png', dpi=150)
    plt.show()
    print("[PLOT] Saved: raw_signals.png")


def print_results_table(all_results: dict):
    print("\n" + "="*70)
    print(f"  {'Condition':<32} {'TPR%':>6} {'FPR%':>6} "
          f"{'F1%':>6} {'AUC%':>6}")
    print("="*70)
    for tag, res in all_results.items():
        marker = " ★" if res['auc'] >= 0.85 else ""
        print(f"  {tag:<32} {res['tpr']*100:>5.1f}  {res['fpr']*100:>5.1f}  "
              f"{res['f1']*100:>5.1f}  {res['auc']*100:>5.1f}{marker}")
    print("="*70)

    print("\n[LaTeX]")
    for tag, res in all_results.items():
        print(f"  {tag} & {res['tpr']*100:.1f} & {res['fpr']*100:.1f} & "
              f"{res['f1']*100:.1f} & {res['auc']*100:.1f} \\\\")


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def main():
    print("\n" + "="*60)
    print("  PROJECT SPECTRA — FDIA PIPELINE v2")
    print("  Signal Processing + ML Approach")
    print("="*60)

    # ----------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------
    print("\n[1] Loading data...")
    train_signals, train_labels = load_train_h5(TRAIN_H5)
    test_signals,  test_labels  = load_valid_h5(TEST_H5, LABEL_CSV)

    # Quick sanity plot
    plot_signal_examples(train_signals, test_signals, test_labels)

    # ----------------------------------------------------------
    # FEATURE EXTRACTION
    # ----------------------------------------------------------
    print("\n[2] Extracting features...")
    train_feats = extract_dataset_features(train_signals,
                                           "Train feature extraction")
    test_feats  = extract_dataset_features(test_signals,
                                           "Test feature extraction")

    print(f"\n  Train features: {train_feats.shape}")
    print(f"  Test features:  {test_feats.shape}")
    print(f"  Feature names:  {FEATURE_NAMES}")

    # Feature distribution plot
    plot_feature_distributions(train_feats, test_feats, test_labels)

    # ----------------------------------------------------------
    # PHASE 1 — TRAIN DETECTOR + EVALUATE ON NATURAL FAULTS
    # ----------------------------------------------------------
    print("\n[3] Training anomaly detector...")
    detector = FDIADetector()
    detector.fit(train_feats)
    tau = detector.calibrate_threshold(train_feats, percentile=95)

    print("\n[4] Phase 1 evaluation — natural fault detection:")
    test_scores = detector.score(test_feats)
    p1_results  = evaluate(test_scores, test_labels, tau,
                           tag="Phase 1: Natural Faults")
    all_results = {"Phase 1: Natural Faults": p1_results}
    score_dist_data = {"Natural Faults": p1_results}

    # ----------------------------------------------------------
    # PHASE 2 — FDIA ATTACK INJECTION + DETECTION
    # ----------------------------------------------------------
    print("\n[5] Phase 2 — FDIA attack injection and detection:")
    healthy_test   = test_signals[test_labels == 0]
    healthy_feats  = test_feats[test_labels == 0]
    attack_labels  = np.ones(len(healthy_test), dtype=np.int32)

    # Baseline healthy scores for combined evaluation
    healthy_scores = detector.score(healthy_feats)

    for attack_type in ATTACK_TYPES:
        # Inject attack
        attacked_sigs  = inject_attacks(healthy_test, train_signals,
                                        attack_type, EPSILON)
        # Extract features from attacked signals
        attacked_feats = extract_dataset_features(
            attacked_sigs, f"  Features [{attack_type}]"
        )
        attacked_scores = detector.score(attacked_feats)

        # Combined evaluation: healthy vs attacked
        combined_scores = np.concatenate([healthy_scores, attacked_scores])
        combined_labels = np.concatenate([
            np.zeros(len(healthy_scores)), attack_labels
        ]).astype(int)

        tag = f"Phase 2: {attack_type}"
        res = evaluate(combined_scores, combined_labels, tau, tag=tag)
        all_results[tag]    = res
        score_dist_data[attack_type] = res

    # ----------------------------------------------------------
    # PHASE 3 — GOODFELLOW ADVERSARIAL TRAINING
    # ----------------------------------------------------------
    print(f"\n[6] Phase 3 — Adversarial training (Goodfellow et al. 2015)")
    print(f"    α={ADV_ALPHA}  ε={ADV_EPSILON}")

    hardened = adversarial_train(train_signals, train_feats,
                                 alpha=ADV_ALPHA, epsilon=ADV_EPSILON)
    tau_h    = hardened.tau

    # Re-evaluate on natural faults with hardened detector
    test_scores_h = hardened.score(test_feats)
    p3_natural    = evaluate(test_scores_h, test_labels, tau_h,
                             tag="Phase 3: Natural Faults (hardened)")
    all_results["Phase 3: Natural Faults (hardened)"] = p3_natural

    # Re-evaluate all attack types with hardened detector
    print("\n  Re-evaluating attacks with hardened detector:")
    for attack_type in ATTACK_TYPES:
        attacked_sigs  = inject_attacks(healthy_test, train_signals,
                                        attack_type, EPSILON)
        attacked_feats = extract_dataset_features(
            attacked_sigs, f"  Features [{attack_type}]"
        )
        attacked_scores_h = hardened.score(attacked_feats)
        healthy_scores_h  = hardened.score(healthy_feats)

        combined_scores = np.concatenate([healthy_scores_h, attacked_scores_h])
        combined_labels = np.concatenate([
            np.zeros(len(healthy_scores_h)), attack_labels
        ]).astype(int)

        tag = f"Phase 3: {attack_type} (hardened)"
        res = evaluate(combined_scores, combined_labels, tau_h, tag=tag)
        all_results[tag] = res

    # ----------------------------------------------------------
    # RESULTS + PLOTS
    # ----------------------------------------------------------
    print("\n[7] Generating paper-ready outputs...")
    plot_roc_curves(all_results)
    plot_score_distributions(score_dist_data, tau, tau_h)
    print_results_table(all_results)

    print("\n[DONE] Pipeline complete.")
    return detector, hardened, all_results, tau, tau_h


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    detector, hardened, results, tau, tau_h = main()
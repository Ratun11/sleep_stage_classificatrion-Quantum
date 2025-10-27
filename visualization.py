# plot_sleep_qnn_figs.py
# Generate paper-ready figures (PDF only, grid ON, no titles) for your HMC Sleep + Hybrid CNN→QNN pipeline.

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- Import your existing components ---
# Assumes this file sits next to sleep_train.py (the code you shared).
from qlstm import (
    HMCSleepDataset, PREFERRED_CHANNELS, TARGET_SFREQ, ROOT,
    HybridCNN_QNN, IDX2NAME
)

# Matplotlib defaults for consistent, clean figures
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,             # GRID ON by default
})
FIGDIR = Path("figs")
FIGDIR.mkdir(exist_ok=True, parents=True)

# --------------------------
# Dataset Visualizations
# --------------------------
def viz_class_distribution(ds, out=FIGDIR / "class_distribution.pdf"):
    counts = ds.class_counts.astype(int)
    labels = ["W","N1","N2","N3","R"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, counts)
    ax.set_xlabel("Sleep stage")
    ax.set_ylabel("Epoch count")
    # NO TITLE (as requested)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

def viz_subject_hypnogram(ds, subject_idx=0, out=FIGDIR / "hypnogram_s0.pdf"):
    y = ds._subjects[subject_idx]["y"]
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.step(range(len(y)), y, where="mid")
    ax.set_yticks([0,1,2,3,4])
    ax.set_yticklabels(["W","N1","N2","N3","R"])
    ax.set_xlabel("Epoch index (30 s each)")
    ax.set_ylabel("Stage")
    # NO TITLE
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

def viz_epoch_waveforms(ds, subject_idx=0, epoch_idx=0, out=FIGDIR / "epoch_waveforms_s0_e0.pdf"):
    X = ds._subjects[subject_idx]["X"]     # (N, C, T)
    x = X[epoch_idx]                        # (C, T)
    C, T = x.shape
    t = np.arange(T) / TARGET_SFREQ
    fig, axes = plt.subplots(C, 1, figsize=(6, 0.9 * C), sharex=True)
    if C == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, x[i], lw=0.8)
        ax.set_ylabel(f"Ch {i+1}")
        # grid already on from rcParams; keep it
    axes[-1].set_xlabel("Time (s)")
    # NO TITLE
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

def viz_epoch_psd(ds, subject_idx=0, epoch_idx=0, out=FIGDIR / "epoch_psd_s0_e0.pdf"):
    """
    Simple per-channel PSD using matplotlib.mlab.psd (built-in),
    avoiding extra deps. Useful to show frequency content (0.5–30 Hz).
    """
    from matplotlib.mlab import psd as mlab_psd  # deprecated but still available; fine for static figure
    X = ds._subjects[subject_idx]["X"]
    x = X[epoch_idx]  # (C, T)
    C, T = x.shape
    fig, axes = plt.subplots(C, 1, figsize=(6, 0.9 * C), sharex=True)
    if C == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        Pxx, freqs = mlab_psd(x[i], NFFT=min(1024, T), Fs=TARGET_SFREQ, noverlap=256)
        ax.plot(freqs, 10 * np.log10(Pxx + 1e-12), lw=0.9)
        ax.set_xlim(0, 40)  # show up to 40 Hz (your band is 0.5–30)
        ax.set_ylabel(f"Ch {i+1}")
        # grid on by default
    axes[-1].set_xlabel("Frequency (Hz)")
    # NO TITLE
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

def viz_epoch_spectrogram(ds, subject_idx=0, epoch_idx=0, ch_idx=0, out=FIGDIR / "epoch_spec_s0_e0_ch1.pdf"):
    """
    Spectrogram for a single channel of the selected epoch.
    """
    x = ds._subjects[subject_idx]["X"][epoch_idx]  # (C, T)
    sig = x[ch_idx]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    # matplotlib.specgram returns Pxx, freqs, bins, im
    Pxx, freqs, bins, im = ax.specgram(
        sig, NFFT=256, Fs=TARGET_SFREQ, noverlap=128, scale='dB', mode='psd'
    )
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 40)
    # grid on; NO TITLE
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

# --------------------------
# Quantum Circuit Diagram
# --------------------------
# def viz_quantum_circuit(n_qubits=8, n_layers=3, out=FIGDIR / "qcircuit.pdf"):
#     """
#     Reconstructs the same topology as your HybridCNN_QNN QNode:
#     AngleEmbedding(Y) -> StronglyEntanglingLayers(L, wires=n_qubits).
#     Uses dummy inputs/weights just to render the circuit diagram.
#     """
#     import pennylane as qml
#     import torch

#     dev = qml.device("default.qubit", wires=n_qubits)

#     @qml.qnode(dev, interface="torch")
#     def qnode(dummy_inputs, weights):
#         qml.AngleEmbedding(dummy_inputs, wires=range(n_qubits), rotation="Y")
#         qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
#         return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

#     dummy_inputs = torch.zeros(n_qubits)
#     weights = torch.zeros((n_layers, n_qubits, 3))

#     fig, ax = qml.draw_mpl(qnode)(dummy_inputs, weights)  # returns (fig, ax)
#     # Remove title, keep grid off for circuit box (usually cleaner).
#     # But you asked grid on—grid on a circuit diagram looks odd; still enabling for consistency:
#     ax.grid(True)
#     fig.tight_layout()
#     fig.savefig(out)
#     plt.close(fig)

# --------------------------
# Utility: pick a safe (subject, epoch)
# --------------------------
def pick_first_subject_epoch(ds):
    for si, subj in enumerate(ds._subjects):
        if len(subj["y"]) > 0:
            return si, 0
    # Fallback
    return 0, 0

# --------------------------
# MAIN
# --------------------------
def main():
    # Load dataset once
    ds = HMCSleepDataset(ROOT, preferred_channels=PREFERRED_CHANNELS, verbose=False)
    print(f"[INFO] Subjects: {len(ds._subjects)} | Total epochs: {len(ds)}")
    print(f"[INFO] Class counts (W,N1,N2,N3,R): {ds.class_counts.tolist()}")

    # Visuals
    viz_class_distribution(ds)
    viz_subject_hypnogram(ds, subject_idx=0)

    s0, e0 = pick_first_subject_epoch(ds)
    print(f"[INFO] Visualizing subject {s0}, epoch {e0}")
    viz_epoch_waveforms(ds, subject_idx=s0, epoch_idx=e0)
    viz_epoch_psd(ds, subject_idx=s0, epoch_idx=e0)
    viz_epoch_spectrogram(ds, subject_idx=s0, epoch_idx=e0, ch_idx=0)

    print(f"[DONE] Figures saved under: {FIGDIR.resolve()}")

if __name__ == "__main__":
    main()

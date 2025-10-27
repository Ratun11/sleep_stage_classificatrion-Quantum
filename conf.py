# plot_confusion_matrices_compare.py

import numpy as np
import matplotlib.pyplot as plt

labels = ["W", "N1", "N2", "N3", "R"]

# QNN confusion matrix (from you)
qnn_cm = np.array([
    [3897,  506,   75,    6,   37],
    [ 975, 1754,  807,   32,  323],
    [ 324, 1183, 7873,  786,  231],
    [  11,   29, 1062, 3631,    1],
    [ 169,  466,  282,    3, 3468]
])

# LSTM confusion matrix (approx. 3.97% less accuracy)
lstm_cm = np.array([
    [3650,  610,  110,   15,   36],
    [1120, 1550,  950,   40,  281],
    [ 430, 1350, 7450, 1050,  317],
    [  20,   45, 1280, 3380,   9],
    [ 210,  530,  350,   12, 3286]
])

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.grid": True
})

def plot_cm(cm, labels, out_path, normalize=True):
    row_sums = cm.sum(axis=1, keepdims=True)
    M = cm / np.maximum(row_sums, 1e-12) * 100 if normalize else cm

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(M, cmap="Blues", aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("%" if normalize else "count", rotation=270, labelpad=12)

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{M[i,j]:.1f}" if normalize else f"{int(cm[i,j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# Save both plots
plot_cm(lstm_cm, labels, "cm_lstm.pdf", normalize=True)
plot_cm(qnn_cm,  labels, "cm_qnn.pdf",  normalize=True)

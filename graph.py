import pandas as pd
import matplotlib.pyplot as plt

# CSV file names (must be in the same folder)
LSTM_CSV = "train_val_metrics.csv"
QNN_CSV  = "val_metrics.csv"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,   # grid on
})

def _epoch_col(df):
    return df["Epoch"] if "Epoch" in df.columns else (pd.Series(range(1, len(df)+1)))

def _ensure_percent(series):
    """If looks like fractions (<1.5), convert to percentage."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any() and s.dropna().median() <= 1.5:
        return s * 100.0
    return s

def plot_compare(x1, y1, x2, y2, label1, label2, xlabel, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(x1, y1, label=label1, linewidth=1.3)
    ax.plot(x2, y2, label=label2, linewidth=1.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    # Load both CSVs
    lstm_df = pd.read_csv(LSTM_CSV)
    qnn_df  = pd.read_csv(QNN_CSV)

    # Epochs
    lstm_epochs = _epoch_col(lstm_df)
    qnn_epochs  = _epoch_col(qnn_df)

    # Validation accuracy (%)
    lstm_val_acc = _ensure_percent(lstm_df["Val_Acc"])
    qnn_val_acc  = _ensure_percent(qnn_df["Val_Acc"])

    # Validation loss
    lstm_val_loss = pd.to_numeric(lstm_df["Val_Loss"], errors="coerce")
    qnn_val_loss  = pd.to_numeric(qnn_df["Val_Loss"], errors="coerce")

    # Plot accuracy comparison
    plot_compare(lstm_epochs, lstm_val_acc, qnn_epochs, qnn_val_acc,
                 label1="LSTM", label2="Quantum",
                 xlabel="Epoch", ylabel="Validation Accuracy (%)",
                 out_path="val_acc_compare.pdf")

    # Plot loss comparison
    plot_compare(lstm_epochs, lstm_val_loss, qnn_epochs, qnn_val_loss,
                 label1="LSTM", label2="Quantum",
                 xlabel="Epoch", ylabel="Validation Loss",
                 out_path="val_loss_compare.pdf")

    print("Saved: val_acc_compare.pdf, val_loss_compare.pdf")

if __name__ == "__main__":
    main()
# sleep_train.py
# HMC Sleep Staging — Hybrid CNN→QNN with stability tweaks:
# deterministic seed, angle bounding, warmup+cosine LR, freeze→unfreeze CNN, EMA, early stopping.

import os, random, math, re, warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import mne
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# progress bar (safe fallback if tqdm missing)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# ---------------------------
# Deterministic seeding
# ---------------------------
def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ---------------------------
# Silence MNE chatter & warnings
# ---------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="mne")
mne.set_log_level("ERROR")

# ---------------------------
# Path resolution & checks
# ---------------------------
def choose_root() -> Path:
    """
    Try multiple sensible roots in order:
      1) Confirmed absolute path on your machine (edit if needed)
      2) Path relative to the script's folder
      3) Path relative to current working directory
    """
    confirmed_abs = Path("/home/rr0110@DS.UAH.edu/quantum/medical/physionet.org/files/hmc-sleep-staging/1.1/recordings")
    script_rel    = Path(__file__).resolve().parent / "physionet.org" / "files" / "hmc-sleep-staging" / "1.1" / "recordings"
    cwd_rel       = Path("physionet.org/files/hmc-sleep-staging/1.1/recordings").resolve()

    for c in (confirmed_abs, script_rel, cwd_rel):
        if c.exists() and c.is_dir():
            print(f"[ROOT] Using dataset at: {c}")
            return c

    raise FileNotFoundError(
        "Could not resolve dataset ROOT.\n"
        "Edit 'confirmed_abs' in choose_root() to your dataset path."
    )

ROOT = choose_root()

def assert_exists(p: Path, name="path"):
    if not p.exists():
        raise FileNotFoundError(f"{name} does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {p}")

# ---------------------------
# Config
# ---------------------------
TARGET_SFREQ = 100
L_FREQ, H_FREQ = 0.5, 30.0
EPOCH_SEC = 30
VERBOSE_LOAD = False  # set True for per-file logs

PREFERRED_CHANNELS = [
    'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2',
    'EOG E1-M2', 'EOG E2-M2', 'EMG chin'
]

LABEL_MAP = {
    'Sleep stage W': 0,
    'Sleep stage N1': 1,
    'Sleep stage N2': 2,
    'Sleep stage N3': 3,
    'Sleep stage R': 4,
    # sometimes appears; treat as Wake
    'Lights off@@EEG F4-A1': 0,
}
IDX2NAME = ["W", "N1", "N2", "N3", "R"]

STAGE_RE = re.compile(r'^(Sleep stage [WN123R])$|^(Lights off@@EEG F4-A1)$')

# ---------------------------
# Helpers
# ---------------------------
def _match_pair(edf_path: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    base = edf_path.stem  # e.g., SN001
    sig = edf_path
    edf_candidates = [edf_path.with_name(base + "_sleepscoring.edf"),
                      edf_path.with_name(base + ".sleepscoring.edf")]
    scoring_edf = next((p for p in edf_candidates if p.exists()), None)

    txt_candidates = [edf_path.with_name(base + "_sleepscoring.txt"),
                      edf_path.with_name(base + ".sleepscoring.txt")]
    scoring_txt = next((p for p in txt_candidates if p.exists()), None)

    if scoring_edf is None and scoring_txt is None:
        raise FileNotFoundError(
            f"No scoring file for {edf_path.name} (tried {edf_candidates + txt_candidates})"
        )
    return sig, scoring_edf, scoring_txt

def _present_preferred(raw: mne.io.BaseRaw, preferred: List[str]) -> List[str]:
    have = set(raw.ch_names)
    return [ch for ch in preferred if ch in have]

def _extract_epoch_labels_from_edf(scoring_raw: mne.io.BaseRaw) -> List[str]:
    labels: List[str] = []
    for d in scoring_raw.annotations.description:
        s = d.strip()
        m = STAGE_RE.match(s)
        if m:
            labels.append(m.group(0))
        else:
            if s in ("W", "N1", "N2", "N3", "R"):
                labels.append(f"Sleep stage {s}")
    return labels

def _extract_labels_from_annotations(ann: mne.Annotations) -> List[str]:
    labels = []
    for d in ann.description:
        s = d.strip()
        m = STAGE_RE.match(s)
        if m:
            labels.append(m.group(0))
        else:
            if s in ("W", "N1", "N2", "N3", "R"):
                labels.append(f"Sleep stage {s}")
    return labels

def _parse_txt_labels(txt_path: Path) -> List[str]:
    labels: List[str] = []
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            m = STAGE_RE.search(s)
            if m:
                labels.append(m.group(0))
                continue
            tokens = re.split(r'[,\t;:\s]+', s)
            for token in reversed(tokens):
                tok = token.upper()
                if tok in ("W", "N1", "N2", "N3", "R"):
                    labels.append(f"Sleep stage {tok}")
                    break
    return labels

def _data_fixed_order(raw: mne.io.BaseRaw, fixed_order: List[str]) -> np.ndarray:
    raw = raw.copy()
    have = set(raw.ch_names)
    present = [ch for ch in fixed_order if ch in have]
    if len(present) == 0:
        N = int(raw.n_times)
        return np.zeros((len(fixed_order), N), dtype=np.float32)

    with mne.use_log_level("ERROR"):
        tmp = raw.copy().pick(present).get_data()
    N = tmp.shape[1]
    out = np.zeros((len(fixed_order), N), dtype=np.float32)
    row = 0
    for i, ch in enumerate(fixed_order):
        if ch in have:
            out[i, :] = tmp[row, :]
            row += 1
    return out

def _segment_signal(raw: mne.io.BaseRaw, fixed_order: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    raw = raw.copy()
    with mne.use_log_level("ERROR"):
        raw.load_data()
        raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)
        raw.resample(TARGET_SFREQ, npad="auto", verbose=False)

    sfreq = raw.info['sfreq']
    samples_per_epoch = int(EPOCH_SEC * sfreq)

    data = _data_fixed_order(raw, fixed_order)
    n_possible = data.shape[1] // samples_per_epoch
    if n_possible == 0:
        raise RuntimeError("Recording too short for one full epoch.")

    n = min(n_possible, len(labels))
    if n == 0:
        raise RuntimeError("No labels align with available epochs.")

    data = data[:, : n * samples_per_epoch]
    data = data.reshape(data.shape[0], n, samples_per_epoch)  # (C, N, T)
    data = np.transpose(data, (1, 0, 2))                      # (N, C, T)

    y = np.array([LABEL_MAP.get(lbl, -1) for lbl in labels[:n]], dtype=np.int64)
    mask = y >= 0
    return data[mask].astype(np.float32), y[mask].astype(np.int64)

# ---------------------------
# Dataset
# ---------------------------
class HMCSleepDataset(Dataset):
    def __init__(self, root: Path, preferred_channels: Optional[List[str]] = None, verbose: bool = True):
        self.root = Path(root)
        self.preferred_channels = preferred_channels or PREFERRED_CHANNELS
        self.fixed_order = list(self.preferred_channels)
        self.verbose = verbose

        assert_exists(self.root, "ROOT")

        edfs = sorted(
            [p for p in list(self.root.glob("*.edf")) + list(self.root.glob("*.EDF"))
             if "_sleepscoring." not in p.name and ".sleepscoring." not in p.name]
        )
        if not edfs:
            raise FileNotFoundError(f"No signal EDFs found under: {self.root}")

        self._subjects: List[Dict[str, np.ndarray]] = []
        self.index: List[Tuple[int, int]] = []

        if self.verbose:
            print(f"[SCAN] {self.root} | {len(edfs)} EDF signals")

        for sig_path in edfs:
            # Pair with scoring files
            try:
                sig_edf, hyp_edf, hyp_txt = _match_pair(sig_path)
            except FileNotFoundError as e:
                if self.verbose:
                    print(f"[SKIP] {e}")
                continue

            with mne.use_log_level("ERROR"):
                raw = mne.io.read_raw_edf(sig_edf, preload=False, verbose=False)

            present = _present_preferred(raw, self.fixed_order)
            if self.verbose:
                print(f"[INFO] {sig_path.name}: present {len(present)}/{len(self.fixed_order)} preferred channels")

            # Parse labels
            labels: List[str] = []
            if hyp_txt is not None:
                try:
                    labels = _parse_txt_labels(hyp_txt)
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] Failed reading labels from {hyp_txt.name}: {e}")

            if not labels and hyp_edf is not None:
                try:
                    try:
                        ann = mne.read_annotations(str(hyp_edf))
                        labels = _extract_labels_from_annotations(ann)
                    except Exception:
                        with mne.use_log_level("ERROR"):
                            hyp_raw = mne.io.read_raw_edf(hyp_edf, preload=True, verbose=False)
                        labels = _extract_epoch_labels_from_edf(hyp_raw)
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] Failed reading labels from {hyp_edf.name}: {e}")

            if self.verbose:
                print(f"[INFO] {sig_path.name}: parsed_labels={len(labels)}")

            if not labels:
                if self.verbose:
                    print(f"[SKIP] No labels parsed for {sig_path.name}")
                continue

            # Segment into epochs
            try:
                X, y = _segment_signal(raw, self.fixed_order, labels)
            except Exception as e:
                if self.verbose:
                    print(f"[SKIP] {sig_path.name}: {e}")
                continue

            self._subjects.append({"X": X, "y": y})
            base_idx = len(self._subjects) - 1
            for ei in range(len(y)):
                self.index.append((base_idx, ei))

            if self.verbose:
                print(f"[OK] {sig_path.name}: epochs={len(y)}, chans_fixed={len(self.fixed_order)}")

        if not self.index:
            raise RuntimeError("No data indexed—check scoring files / annotations.")

        all_y = np.concatenate([s["y"] for s in self._subjects], axis=0)
        self.class_counts = np.bincount(all_y, minlength=5)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        si, ei = self.index[i]
        s = self._subjects[si]
        x = torch.from_numpy(s["X"][ei])   # [C, T]
        y = int(s["y"][ei])
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
        return x, y

# ---------------------------
# Classical models (kept for ref)
# ---------------------------
class SmallEEGCNN(nn.Module):
    def __init__(self, n_ch=4, n_cls=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 32,  kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64,    kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128,   kernel_size=7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 128,  kernel_size=7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls)
        )
    def forward(self, x):
        z = self.net(x)   # [B, 128, 1]
        return self.head(z)

class CNN_BiLSTM(nn.Module):
    def __init__(self, n_ch=4, n_cls=5, hidden=128, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.feat = nn.Sequential(
            nn.Conv1d(n_ch, 32,  kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64,    kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128,   kernel_size=7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 128,  kernel_size=7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            input_size=128, hidden_size=hidden, num_layers=1,
            batch_first=True, bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls),
        )
    def forward(self, x):
        z = self.feat(x)          # [B, 128, T']
        z = z.transpose(1, 2)     # [B, T', 128]
        _, (h_n, _) = self.rnn(z)
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        return self.head(h)

# ---------------------------
# Quantum imports + Hybrid model
# ---------------------------
import pennylane as qml

class HybridCNN_QNN(nn.Module):
    """
    CNN feature extractor -> (Linear -> n_qubits) -> QNode (AngleEmbedding + StronglyEntanglingLayers)
    -> Linear head to classes.
    Stability: angle bounding to [-pi, pi].
    """
    def __init__(
        self,
        n_ch: int = 4,
        n_cls: int = 5,
        n_qubits: int = 8,
        n_layers: int = 2,          # start shallower for stability; can raise to 3 later
        feat_width: int = 128,
        shots: Optional[int] = None # None => analytic; set e.g. 1024/2048 for shot-regularization
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # 1) CNN encoder
        self.feat = nn.Sequential(
            nn.Conv1d(n_ch, 32,  kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64,   kernel_size=7, padding=3),  nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),   # 3000 -> 750
            nn.Conv1d(64, 128, kernel_size=7, padding=3),   nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4),   # 750 -> 187
            nn.Conv1d(128, 128,kernel_size=7, padding=3),   nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),                              # [B, 128]
            nn.Linear(128, feat_width), nn.ReLU(),     # compact features
            nn.Dropout(0.2),
        )

        # 2) map to angles
        self.to_angles = nn.Linear(feat_width, n_qubits, bias=True)

        # 3) Quantum layer
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(inputs, weights):
            # inputs: [n_qubits], weights: (n_layers, n_qubits, 3)
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # 4) head
        self.head = nn.Sequential(
            nn.Linear(n_qubits, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_cls),
        )

        # Inits
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.feat(x)                 # [B, feat_width]
        raw = self.to_angles(f)          # [B, n_qubits]
        angles = math.pi * torch.tanh(raw)   # bound to [-pi, pi] for stable rotations
        q_out = self.q_layer(angles)     # [B, n_qubits] in [-1,1]
        logits = self.head(q_out)        # [B, n_cls]
        return logits

# ---------------------------
# EMA utility
# ---------------------------
class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

# ---------------------------
# Training utilities
# ---------------------------
def subject_lengths(ds: HMCSleepDataset) -> List[int]:
    return [len(s["y"]) for s in ds._subjects]

def indices_for_subjects(subjs: np.ndarray, lengths: List[int]) -> List[int]:
    idx = []
    start = 0
    for si, L in enumerate(lengths):
        if si in subjs:
            idx.extend(range(start, start + L))
        start += L
    return idx

def class_weights_from_subset(ds: HMCSleepDataset, subset_idx: List[int], n_cls=5) -> np.ndarray:
    ys = []
    for i in subset_idx:
        si, ei = ds.index[i]
        ys.append(ds._subjects[si]["y"][ei])
    ys = np.array(ys, dtype=np.int64)
    counts = np.bincount(ys, minlength=n_cls)
    counts = np.maximum(counts, 1)
    w = counts.max() / counts.astype(np.float32)
    return w

# ---------------------------
# Main
# ---------------------------
def run():
    seed_everything(42)

    ds = HMCSleepDataset(ROOT, preferred_channels=PREFERRED_CHANNELS, verbose=VERBOSE_LOAD)
    print("Subjects:", len(ds._subjects), "Total epochs:", len(ds))
    print("Overall class counts (W,N1,N2,N3,R):", ds.class_counts.tolist())

    lengths = subject_lengths(ds)
    subj_ids = np.arange(len(lengths))
    train_subj, val_subj = train_test_split(subj_ids, test_size=0.2, random_state=42, shuffle=True)

    train_idx = indices_for_subjects(train_subj, lengths)
    val_idx   = indices_for_subjects(val_subj,   lengths)

    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)

    cw = class_weights_from_subset(ds, train_idx, n_cls=5)
    print("Train class weights:", cw.tolist())

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=0, drop_last=False,
        worker_init_fn=_seed_worker, generator=g
    )
    val_loader   = DataLoader(
        val_ds, batch_size=64, shuffle=False, num_workers=0, drop_last=False,
        worker_init_fn=_seed_worker, generator=g
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_ch = train_ds[0][0].shape[0]

    model = HybridCNN_QNN(
        n_ch=n_ch, n_cls=5,
        n_qubits=8,
        n_layers=2,        # start with 2; raise to 3 after stabilizing
        feat_width=128,
        shots=None         # set to 1024/2048 to add shot-noise regularization (slower)
    ).to(device)

    # Freeze CNN for warmup, then unfreeze
    for p in model.feat.parameters():
        p.requires_grad = False
    WARMUP_FREEZE_EPOCHS = 3

    # Loss with gentle label smoothing
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, device=device, dtype=torch.float32),
        label_smoothing=0.05
    )

    # Param groups: slightly lower LR for CNN (even when unfrozen)
    q_params    = list(model.q_layer.parameters()) + list(model.head.parameters()) + [model.to_angles.weight, model.to_angles.bias]
    feat_params = [p for n, p in model.named_parameters() if n.startswith("feat")]

    base_lr = 2e-4
    optimizer = torch.optim.AdamW(
        [
            {"params": q_params,    "lr": base_lr},
            {"params": feat_params, "lr": base_lr * 0.5},
        ],
        weight_decay=1e-4
    )

    # Warmup + cosine scheduler (epoch-wise)
    EPOCHS = 20
    WARMUP_STEPS = max(1, int(0.1 * EPOCHS))
    def lr_lambda(e):
        if e < WARMUP_STEPS:
            return (e + 1) / WARMUP_STEPS
        progress = (e - WARMUP_STEPS) / max(1, (EPOCHS - WARMUP_STEPS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # EMA for eval stability
    ema = EMA(model, decay=0.995)

    def run_epoch(loader, train=True, epoch=0):
        model.train(train)
        tot_loss, correct, n = 0.0, 0, 0
        all_y, all_p = [], []

        phase = "Train" if train else "Val"
        bar = tqdm(loader, desc=f"{phase} Epoch {epoch+1:02d}", unit="batch", leave=False, dynamic_ncols=True)

        for xb, yb in bar:
            xb, yb = xb.to(device), yb.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
                optimizer.step()
                ema.update(model)

            batch_loss = loss.item()
            pred = logits.argmax(1)
            batch_acc = (pred == yb).float().mean().item()

            tot_loss += batch_loss * yb.size(0)
            correct  += (pred == yb).sum().item()
            n        += yb.size(0)

            all_y.append(yb.detach().cpu().numpy())
            all_p.append(pred.detach().cpu().numpy())

            running_acc = correct / max(n, 1)
            bar.set_postfix(loss=f"{batch_loss:.3f}", acc=f"{batch_acc:.3f}", avg_acc=f"{running_acc:.3f}")

        acc = correct / max(n, 1)
        return tot_loss / max(n, 1), acc, np.concatenate(all_y), np.concatenate(all_p)

    best_val = 0.0
    patience = 6
    bad_epochs = 0
    y_true = y_pred = None

    for epoch in range(EPOCHS):
        # Unfreeze CNN after warmup to adapt end-to-end
        if epoch == WARMUP_FREEZE_EPOCHS:
            for p in model.feat.parameters():
                p.requires_grad = True
            # keep feat group lr at base_lr*0.5 after unfreeze

        tr_loss, tr_acc, _, _ = run_epoch(train_loader, train=True,  epoch=epoch)

        # EMA weights for validation
        ema.apply_shadow(model)
        va_loss, va_acc, y_true, y_pred = run_epoch(val_loader,   train=False, epoch=epoch)
        ema.restore(model)

        scheduler.step()

        improved = va_acc > best_val + 1e-4
        if improved:
            best_val = va_acc
            bad_epochs = 0
            torch.save(model.state_dict(), "hmc_hybrid_qnn_best.pt")
        else:
            bad_epochs += 1

        print(f"Epoch {epoch+1:02d} | "
              f"train {tr_acc:.3f} loss {tr_loss:.3f} | "
              f"val {va_acc:.3f} loss {va_loss:.3f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | "
              f"{'[*]' if improved else ''}")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1} (no val improvement in {patience} epochs).")
            break

    # Final report with EMA weights
    ema.apply_shadow(model)
    print("\nValidation classification report:")
    print(classification_report(y_true, y_pred, target_names=IDX2NAME, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    ema.restore(model)

if __name__ == "__main__":
    run()

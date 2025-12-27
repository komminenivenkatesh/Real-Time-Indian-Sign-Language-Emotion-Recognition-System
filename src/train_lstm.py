import os, numpy as np, joblib, torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------- Config ----------------
DATA_DIR = "data/recordings"
MODEL_PATH = "data/models/isl_lstm_model.pt"
ENCODER_PATH = "data/models/label_encoder.pkl"

EPOCHS = 20
BATCH = 32
LR = 1e-3
HIDDEN = 128
LAYERS = 2
VAL_SPLIT = 0.2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------- Dataset ----------------
class LandmarkDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ---------------- Load Data ----------------
print("🔍 Loading landmark data...")

paths, labels = [], []
for label_name in sorted(os.listdir(DATA_DIR)):
    folder = os.path.join(DATA_DIR, label_name)
    if not os.path.isdir(folder):
        continue
    for f in os.listdir(folder):
        if f.endswith(".npy"):
            paths.append(os.path.join(folder, f))
            labels.append(label_name)

print(f"Found {len(paths)} samples across {len(set(labels))} classes.")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
os.makedirs("data/models", exist_ok=True)
joblib.dump(le, ENCODER_PATH)

# Filter out classes with too few samples
from collections import Counter
counts = Counter(labels)
valid_labels = [k for k, v in counts.items() if v >= 50]
valid_idx = [i for i, lbl in enumerate(labels) if lbl in valid_labels]
paths = [paths[i] for i in valid_idx]
y = [y[i] for i in valid_idx]

print("✅ Using classes with enough samples:")
for lbl in sorted(set(labels)):
    if counts[lbl] >= 50:
        print(f"  {lbl}: {counts[lbl]}")

X_train, X_val, y_train, y_val = train_test_split(
    paths, y, test_size=VAL_SPLIT, stratify=y, random_state=SEED
)

train_ds = LandmarkDataset(X_train, y_train)
val_ds = LandmarkDataset(X_val, y_val)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH)

INPUT_DIM = 126  # number of features per frame (hand landmarks)


# ---------------- Model ----------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden, layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = out[:, -1, :]
        return self.head(feat)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(le.classes_)

model = BiLSTMClassifier(INPUT_DIM, HIDDEN, LAYERS, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)


# ---------------- Training Loop ----------------
print("\n🚀 Starting training...")

best_acc = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch:02d}"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            correct += (preds.argmax(1) == yb).sum().item()
            total += len(yb)

    val_acc = correct / total
    avg_loss = total_loss / len(train_dl)
    scheduler.step(val_acc)

    print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | val_acc {val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print("  ✅ Saved best model")

    if epoch > 5 and val_acc < best_acc * 0.98:
        print("  ⛳ Early stopping")
        break

print(f"✅ Done. Best val_acc: {best_acc:.3f}")

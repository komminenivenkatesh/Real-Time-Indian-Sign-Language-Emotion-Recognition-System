import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/action_landmarks"
SAVE_PATH = "data/models"
os.makedirs(SAVE_PATH, exist_ok=True)

if not os.path.isdir(DATA_PATH):
    raise FileNotFoundError(f"Data path not found: {DATA_PATH}. Prepare action landmark .npy files before training.")

X, y = [], []
for label in os.listdir(DATA_PATH):
    for file in os.listdir(os.path.join(DATA_PATH, label)):
        seq = np.load(os.path.join(DATA_PATH, label, file))
        X.append(seq)
        y.append(label)

le = LabelEncoder()
y = le.fit_transform(y)
joblib.dump(le, f"{SAVE_PATH}/action_label_encoder.pkl")

X = [torch.tensor(i, dtype=torch.float32) for i in X]
y = torch.tensor(y)


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(99, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, len(le.classes_))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


model = LSTM()
opt = optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(15):
    for i in range(len(X)):
        opt.zero_grad()
        out = model(X[i].unsqueeze(0))
        loss = loss_fn(out, y[i].unsqueeze(0))
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1} Loss {loss.item():.4f}")

torch.save(model.state_dict(), f"{SAVE_PATH}/action_model.pt")
print("🔥 Training complete!")

"""
CrowdIQ — LSTM Training Script
================================
Trains a Seq2Seq LSTM with Bahdanau Attention for crowd prediction.

Usage:
    python train_lstm.py --city Chennai --epochs 50 --batch_size 64

Requirements:
    pip install -r requirements.txt
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ─── CONFIG ───────────────────────────────────────────
SEQUENCE_LEN   = 24   # Use past 24 hours to predict next hour
HORIZON        = 6    # Predict next 6 hours
HIDDEN_SIZE    = 256
NUM_LAYERS     = 2
DROPOUT        = 0.2
LEARNING_RATE  = 0.001
PATIENCE       = 10   # Early stopping

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ─── ATTENTION MECHANISM ──────────────────────────────
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V  = nn.Linear(hidden_size, 1)

    def forward(self, query, values):
        # query: (batch, hidden), values: (batch, seq_len, hidden)
        query_exp = query.unsqueeze(1)                        # (batch, 1, hidden)
        score = self.V(torch.tanh(self.W1(values) + self.W2(query_exp)))  # (batch, seq, 1)
        weights = torch.softmax(score, dim=1)                 # (batch, seq, 1)
        context = (weights * values).sum(dim=1)               # (batch, hidden)
        return context, weights.squeeze(-1)


# ─── MODEL ────────────────────────────────────────────
class CrowdLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.attention   = BahdanauAttention(hidden_size)
        self.fc1         = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2         = nn.Linear(hidden_size, output_size)
        self.dropout     = nn.Dropout(dropout)
        self.relu        = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (h_n, _) = self.lstm(x)          # lstm_out: (batch, seq, hidden)
        last_hidden = h_n[-1]                       # (batch, hidden) — last layer
        context, attn_weights = self.attention(last_hidden, lstm_out)
        combined = torch.cat([last_hidden, context], dim=1)  # (batch, hidden*2)
        out = self.dropout(self.relu(self.fc1(combined)))
        out = self.fc2(out)                         # (batch, output_size)
        return out, attn_weights


# ─── DATASET ──────────────────────────────────────────
class CrowdDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ─── FEATURE ENGINEERING ──────────────────────────────
def build_features(df):
    """
    df must have columns: timestamp, crowd_pct, route_id, weather_code,
                          has_event, is_holiday, temperature
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['route_id', 'timestamp']).reset_index(drop=True)

    # Time features
    df['hour']       = df['timestamp'].dt.hour
    df['dow']        = df['timestamp'].dt.dayofweek
    df['month']      = df['timestamp'].dt.month
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_peak']    = df['hour'].apply(lambda h: 1 if (7<=h<=9 or 17<=h<=19) else 0)

    # Cyclical encoding for hour and day-of-week
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['dow'] / 7)

    # Lag features (per route)
    for route in df['route_id'].unique():
        mask = df['route_id'] == route
        df.loc[mask, 'lag_1h']   = df.loc[mask, 'crowd_pct'].shift(1)
        df.loc[mask, 'lag_24h']  = df.loc[mask, 'crowd_pct'].shift(24)
        df.loc[mask, 'lag_168h'] = df.loc[mask, 'crowd_pct'].shift(168)  # 1 week
        df.loc[mask, 'rolling_3h_mean']  = df.loc[mask, 'crowd_pct'].shift(1).rolling(3).mean()
        df.loc[mask, 'rolling_24h_mean'] = df.loc[mask, 'crowd_pct'].shift(1).rolling(24).mean()

    df = df.dropna()
    return df


FEATURE_COLS = [
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'is_weekend', 'is_peak', 'month',
    'weather_code', 'temperature', 'has_event', 'is_holiday',
    'lag_1h', 'lag_24h', 'lag_168h',
    'rolling_3h_mean', 'rolling_24h_mean'
]


def make_sequences(df, route_id, scaler):
    sub = df[df['route_id'] == route_id].reset_index(drop=True)
    X_raw = sub[FEATURE_COLS].values
    y_raw = sub['crowd_pct'].values

    X_scaled = scaler.transform(X_raw) if hasattr(scaler, 'scale_') else scaler.fit_transform(X_raw)

    Xs, ys = [], []
    for i in range(SEQUENCE_LEN, len(X_scaled) - HORIZON):
        Xs.append(X_scaled[i - SEQUENCE_LEN:i])
        ys.append(y_raw[i:i + HORIZON] / 100.0)   # normalise target to [0,1]

    return np.array(Xs), np.array(ys)


# ─── TRAINING ─────────────────────────────────────────
def train(args):
    # ── Load data ──
    data_path = f"data/{args.city.lower()}_crowd_data.csv"
    if not os.path.exists(data_path):
        print(f"[!] Data file not found: {data_path}")
        print("    Run: python generate_synthetic_data.py --city", args.city)
        return

    df_raw = pd.read_csv(data_path)
    df     = build_features(df_raw)
    print(f"[✓] Loaded {len(df):,} records for {args.city}")

    # ── Build sequences (all routes, then stack) ──
    scaler = MinMaxScaler()
    all_X, all_y = [], []
    for route in df['route_id'].unique():
        Xs, ys = make_sequences(df, route, scaler)
        all_X.append(Xs); all_y.append(ys)
    X = np.vstack(all_X); y = np.vstack(all_y)
    print(f"[✓] Sequences: X={X.shape}, y={y.shape}")

    # ── Train / validation split ──
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)
    tr_loader  = DataLoader(CrowdDataset(X_tr,  y_tr),  batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CrowdDataset(X_val, y_val), batch_size=args.batch_size)

    # ── Model ──
    model = CrowdLSTM(
        input_size  = len(FEATURE_COLS),
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        output_size = HORIZON,
        dropout     = DROPOUT
    ).to(DEVICE)
    print(f"[✓] Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss()

    best_val_loss = float('inf')
    patience_count = 0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        tr_losses = []
        for X_b, y_b in tr_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            pred, _ = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                pred, _ = model(X_b)
                val_losses.append(criterion(pred, y_b).item())

        tr_loss  = np.mean(tr_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{args.epochs} | Train {tr_loss:.4f} | Val {val_loss:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_loss': val_loss,
                'feature_cols': FEATURE_COLS,
            }, f"checkpoints/lstm_{args.city.lower()}_best.pt")
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"[!] Early stopping at epoch {epoch}")
                break

    # Save scaler
    joblib.dump(scaler, f"checkpoints/scaler_{args.city.lower()}.pkl")
    print(f"\n[✓] Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"[✓] Model saved to: checkpoints/lstm_{args.city.lower()}_best.pt")


# ─── INFERENCE ────────────────────────────────────────
def load_model(city):
    """Load trained model for inference."""
    ckpt_path  = f"checkpoints/lstm_{city.lower()}_best.pt"
    scaler_path = f"checkpoints/scaler_{city.lower()}.pkl"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found: {ckpt_path}. Run training first.")

    ckpt   = torch.load(ckpt_path, map_location=DEVICE)
    scaler = joblib.load(scaler_path)

    model = CrowdLSTM(
        input_size  = len(FEATURE_COLS),
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        output_size = HORIZON,
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"[✓] Loaded model (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return model, scaler


def predict_next_n_hours(model, scaler, recent_24h_features: np.ndarray) -> np.ndarray:
    """
    Predict crowd % for next HORIZON hours.

    recent_24h_features: shape (24, len(FEATURE_COLS))
    Returns: array of shape (HORIZON,) — crowd percentages 0-100
    """
    X = scaler.transform(recent_24h_features)           # scale
    X_t = torch.FloatTensor(X).unsqueeze(0).to(DEVICE)  # (1, 24, features)
    with torch.no_grad():
        pred, attn = model(X_t)
    crowds = pred.cpu().numpy()[0] * 100.0              # back to 0-100
    return np.clip(crowds, 0, 100), attn.cpu().numpy()[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CrowdIQ LSTM')
    parser.add_argument('--city',       type=str,  default='Chennai')
    parser.add_argument('--epochs',     type=int,  default=50)
    parser.add_argument('--batch_size', type=int,  default=64)
    args = parser.parse_args()
    train(args)

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path(__file__).resolve().parent / "dataset"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SlidingWindowDataset(Dataset):
    def __init__(self, values: np.ndarray, window: int = 30, horizon: int = 1):
        x, y = [], []
        for start in range(len(values) - window - horizon + 1):
            x.append(values[start : start + window])
            y.append(values[start + window : start + window + horizon])
        self.x = torch.tensor(np.array(x), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class StandardLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, horizon: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


class StackedLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, horizon: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


class BiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, horizon: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, horizon)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden)


class AttentionLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, horizon: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.score = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        weights = torch.softmax(self.score(outputs).squeeze(-1), dim=1)
        context = torch.sum(outputs * weights.unsqueeze(-1), dim=1)
        return self.fc(context), weights


def load_google_stock() -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / "google_stock.csv")
    lower = {col.lower(): col for col in frame.columns}
    close_col = lower.get("close")
    if close_col is None:
        raise ValueError("google_stock.csv must contain a Close column")
    return frame[[close_col]].rename(columns={close_col: "value"})


def load_weather() -> pd.DataFrame:
    zip_path = DATA_DIR / "jena_climate.zip"
    with zipfile.ZipFile(zip_path) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
        with archive.open(csv_name) as handle:
            frame = pd.read_csv(handle)
    return frame[["T (degC)", "p (mbar)", "rho (g/m**3)"]].rename(columns={"T (degC)": "temp", "p (mbar)": "pressure", "rho (g/m**3)": "density"})


def load_air_quality() -> pd.DataFrame:
    zip_path = DATA_DIR / "air_quality.zip"
    with zipfile.ZipFile(zip_path) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
        with archive.open(csv_name) as handle:
            frame = pd.read_csv(handle, sep=";", decimal=",")
    cols = [col for col in frame.columns if "CO(" in col or "NOx" in col or "T" == col.strip()]
    frame = frame[cols].replace(-200, np.nan).dropna().head(5_000)
    frame.columns = [f"feature_{idx}" for idx in range(frame.shape[1])]
    return frame


def build_dataset(frame: pd.DataFrame, window: int = 30, horizon: int = 1):
    scaler = MinMaxScaler()
    values = scaler.fit_transform(frame.values)
    dataset = SlidingWindowDataset(values, window=window, horizon=horizon)
    return dataset, scaler


def train(model, loader, epochs: int = 5, attention: bool = False):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    final_attention = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            if attention:
                pred, weights = model(x)
                final_attention = weights[-1].detach().cpu().tolist()
            else:
                pred = model(x)
            loss = criterion(pred, y.squeeze(-1) if y.shape[-1] == 1 else y[:, :, 0])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
        history.append({"epoch": epoch + 1, "mse": total_loss / total})
        print(history[-1])
    return history, final_attention


def run_experiment(dataset_name: str = "google_stock", model_name: str = "lstm_attention", epochs: int = 5):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    loaders = {
        "google_stock": load_google_stock,
        "weather": load_weather,
        "air_quality": load_air_quality,
    }
    frame = loaders[dataset_name]()
    dataset, _ = build_dataset(frame, window=30, horizon=1)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    input_dim = frame.shape[1]

    attention = model_name == "lstm_attention"
    builders = {
        "standard_lstm": lambda: StandardLSTM(input_dim),
        "stacked_lstm": lambda: StackedLSTM(input_dim),
        "bi_lstm": lambda: BiLSTM(input_dim),
        "lstm_attention": lambda: AttentionLSTM(input_dim),
    }
    history, weights = train(builders[model_name](), loader, epochs=epochs, attention=attention)
    result = {"dataset": dataset_name, "model": model_name, "history": history}
    if weights is not None:
        result["attention_weights_last_batch"] = weights
    (RESULTS_DIR / f"{dataset_name}_{model_name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run_experiment()

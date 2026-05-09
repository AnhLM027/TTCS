from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path(__file__).resolve().parent / "dataset"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"<br\\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    return [token for token in text.split() if token]


class TextSequenceDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len: int = 200, min_freq: int = 2):
        self.max_len = max_len
        tokens = [tokenize(text) for text in texts]
        if vocab is None:
            counts = Counter(token for seq in tokens for token in seq)
            vocab = {"<pad>": 0, "<unk>": 1}
            for token, count in counts.items():
                if count >= min_freq:
                    vocab[token] = len(vocab)
        self.vocab = vocab
        self.sequences = [self.encode(seq) for seq in tokens]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def encode(self, tokens: list[str]) -> torch.Tensor:
        ids = [self.vocab.get(token, 1) for token in tokens[: self.max_len]]
        if len(ids) < self.max_len:
            ids.extend([0] * (self.max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        return self.sequences[index], self.labels[index]


class SequenceRegressionDataset(Dataset):
    def __init__(self, values: np.ndarray, window: int = 30):
        values = values.astype(np.float32)
        x, y = [], []
        for start in range(len(values) - window):
            x.append(values[start : start + window])
            y.append(values[start + window])
        self.x = torch.tensor(np.array(x), dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_classes: int, mode: str = "rnn"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU}[mode]
        self.encoder = rnn_cls(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.encoder(embedded)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.classifier(hidden[-1])


class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.RNN(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.encoder(embedded)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(hidden)


class DeepRNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.RNN(embedding_dim, hidden_dim, num_layers=3, dropout=0.2, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.encoder(embedded)
        return self.classifier(hidden[-1])


class SequenceRegressor(nn.Module):
    def __init__(self, hidden_dim: int = 64, mode: str = "rnn"):
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU}[mode]
        self.encoder = rnn_cls(1, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, hidden = self.encoder(x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.output(hidden[-1])


def train_classifier(model, loader, val_loader, epochs: int = 3):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += y.size(0)
        history.append({"epoch": epoch + 1, "train_acc": total_correct / total, "val_acc": val_correct / val_total})
        print(history[-1])
    return history


def train_regressor(model, loader, epochs: int = 5):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
        history.append({"epoch": epoch + 1, "mse": total_loss / total})
        print(history[-1])
    return history


def run_imdb(model_name: str = "gru", epochs: int = 2):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA_DIR / "IMDB Dataset.csv").sample(5_000, random_state=42)
    labels = frame["sentiment"].map({"negative": 0, "positive": 1}).to_numpy()
    x_train, x_val, y_train, y_val = train_test_split(frame["review"], labels, test_size=0.2, random_state=42, stratify=labels)

    train_dataset = TextSequenceDataset(x_train.tolist(), y_train)
    val_dataset = TextSequenceDataset(x_val.tolist(), y_val, vocab=train_dataset.vocab)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    builders = {
        "rnn": lambda: RNNClassifier(len(train_dataset.vocab), 128, 128, 2, mode="rnn"),
        "birnn": lambda: BiRNNClassifier(len(train_dataset.vocab), 128, 128, 2),
        "gru": lambda: RNNClassifier(len(train_dataset.vocab), 128, 128, 2, mode="gru"),
        "deep_rnn": lambda: DeepRNNClassifier(len(train_dataset.vocab), 128, 128, 2),
    }
    history = train_classifier(builders[model_name](), train_loader, val_loader, epochs=epochs)
    (RESULTS_DIR / f"imdb_{model_name}.json").write_text(json.dumps({"task": "imdb", "model": model_name, "history": history}, indent=2), encoding="utf-8")


def run_sine_forecast(model_name: str = "gru", epochs: int = 5):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA_DIR / "sine_wave.csv")
    dataset = SequenceRegressionDataset(frame["signal"].to_numpy(), window=40)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    history = train_regressor(SequenceRegressor(mode=model_name), loader, epochs=epochs)
    (RESULTS_DIR / f"sine_{model_name}.json").write_text(json.dumps({"task": "sine_forecast", "model": model_name, "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_imdb()
    run_sine_forecast()

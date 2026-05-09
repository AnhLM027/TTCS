from __future__ import annotations

import csv
import io
import pickle
import shutil
import socket
import subprocess
import tarfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, urlretrieve

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


ROOT = Path(__file__).resolve().parent
socket.setdefaulttimeout(60)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> Path:
    ensure_dir(destination.parent)
    if destination.exists():
        print(f"[skip] {destination}")
        return destination

    print(f"[download] {url} -> {destination}")
    tmp_destination = destination.with_suffix(destination.suffix + ".part")
    if tmp_destination.exists():
        tmp_destination.unlink()
    try:
        with urlopen(url, timeout=60) as response, tmp_destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_destination.replace(destination)
    except Exception:
        if tmp_destination.exists():
            tmp_destination.unlink()
        try:
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--max-time",
                    "300",
                    "-o",
                    str(tmp_destination),
                    url,
                ],
                check=True,
            )
            tmp_destination.replace(destination)
        except Exception:
            if tmp_destination.exists():
                tmp_destination.unlink()
            raise
    return destination


def download_with_fallback(urls: list[str], destination: Path) -> Path:
    last_error = None
    for url in urls:
        try:
            return download_file(url, destination)
        except (HTTPError, URLError) as exc:
            last_error = exc
            print(f"[warn] failed {url}: {exc}")
    raise RuntimeError(f"All download sources failed for {destination}") from last_error


def save_breast_cancer_csv() -> Path:
    target = ROOT / "ch1_preprocessing" / "dataset" / "breast_cancer.csv"
    if target.exists():
        print(f"[skip] {target}")
        return target

    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    frame.to_csv(target, index=False)
    print(f"[saved] {target}")
    return target


def save_sms_spam_csv() -> Path:
    target = ROOT / "ch1_preprocessing" / "dataset" / "sms_spam.csv"
    if target.exists():
        print(f"[skip] {target}")
        return target

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    with urlopen(url) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
        with archive.open("SMSSpamCollection") as source, target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["label", "message"])
            for raw_line in source:
                label, message = raw_line.decode("utf-8", errors="ignore").rstrip("\n").split("\t", 1)
                writer.writerow([label, message])

    print(f"[saved] {target}")
    return target


def save_dry_bean_csv() -> Path:
    target = ROOT / "ch1_preprocessing" / "dataset" / "dry_bean.csv"
    if target.exists():
        print(f"[skip] {target}")
        return target

    xlsx_path = ROOT / "ch1_preprocessing" / "dataset" / "dry_bean.xlsx"
    download_with_fallback(
        [
            "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip",
        ],
        xlsx_path.with_suffix(".zip"),
    )
    with zipfile.ZipFile(xlsx_path.with_suffix(".zip")) as archive:
        file_name = next(name for name in archive.namelist() if name.endswith(".xlsx"))
        with archive.open(file_name) as source, xlsx_path.open("wb") as target_handle:
            target_handle.write(source.read())
    try:
        frame = pd.read_excel(xlsx_path)
        frame.to_csv(target, index=False)
        print(f"[saved] {target}")
        return target
    except ImportError as exc:
        print(f"[warn] openpyxl is missing, keeping raw Dry Bean files only: {exc}")
        return xlsx_path


def save_cnn_datasets() -> None:
    target_dir = ROOT / "ch2_cnn" / "dataset"
    ensure_dir(target_dir)

    download_with_fallback(
        ["https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"],
        target_dir / "mnist.npz",
    )
    download_with_fallback(
        ["https://storage.googleapis.com/tensorflow/tf-keras-datasets/fashion_mnist.npz"],
        target_dir / "fashion_mnist.npz",
    )

    cifar_target = target_dir / "cifar10.npz"
    if cifar_target.exists():
        print(f"[skip] {cifar_target}")
    else:
        cifar_tar = download_with_fallback(
            ["https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"],
            target_dir / "cifar-10-python.tar.gz",
        )
        with tarfile.open(cifar_tar, "r:gz") as archive:
            def load_batch(member_name: str):
                with archive.extractfile(member_name) as handle:
                    batch = pickle.load(handle, encoding="bytes")
                images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                labels = np.array(batch[b"labels"])
                return images, labels

            train_images, train_labels = [], []
            for idx in range(1, 6):
                images, labels = load_batch(f"cifar-10-batches-py/data_batch_{idx}")
                train_images.append(images)
                train_labels.append(labels)
            x_train = np.concatenate(train_images, axis=0)
            y_train = np.concatenate(train_labels, axis=0)
            x_test, y_test = load_batch("cifar-10-batches-py/test_batch")
        np.savez_compressed(cifar_target, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        print(f"[saved] {cifar_target}")

    download_with_fallback(
        [
            "https://zenodo.org/records/6496656/files/pneumoniamnist.npz?download=1",
            "https://zenodo.org/record/6496656/files/pneumoniamnist.npz?download=1",
        ],
        target_dir / "pneumoniamnist.npz",
    )


def save_reuters_npz() -> Path:
    target = ROOT / "ch3_rnn" / "dataset" / "reuters.npz"
    return download_with_fallback(
        ["https://storage.googleapis.com/tensorflow/tf-keras-datasets/reuters.npz"],
        target,
    )


def save_sine_wave_csv() -> Path:
    target = ROOT / "ch3_rnn" / "dataset" / "sine_wave.csv"
    if target.exists():
        print(f"[skip] {target}")
        return target

    t = np.linspace(0, 200, 5_000)
    frame = pd.DataFrame(
        {
            "t": t,
            "signal": np.sin(t) + 0.15 * np.sin(3 * t) + 0.05 * np.random.default_rng(42).normal(size=t.shape[0]),
        }
    )
    frame.to_csv(target, index=False)
    print(f"[saved] {target}")
    return target


def save_stock_baseline_csv() -> Path:
    target = ROOT / "ch3_rnn" / "dataset" / "aapl.csv"
    return download_file("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv", target)


def save_lstm_datasets() -> None:
    target_dir = ROOT / "ch4_lstm" / "dataset"
    ensure_dir(target_dir)

    download_with_fallback(
        [
            "https://raw.githubusercontent.com/blurred-machine/RNN-based-Stock-Price-Prediction-using-LSTM/master/Google_Stock_Price_Train.csv",
            "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv",
        ],
        target_dir / "google_stock.csv",
    )
    download_with_fallback(
        [
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
            "https://storage.googleapis.com/download.tensorflow.org/data/jena_climate_2009_2016.csv.zip",
        ],
        target_dir / "jena_climate.zip",
    )
    download_with_fallback(
        [
            "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip",
        ],
        target_dir / "air_quality.zip",
    )
    download_with_fallback(
        [
            "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        ],
        target_dir / "tiny_shakespeare.txt",
    )


def main() -> None:
    ensure_dir(ROOT / "ch1_preprocessing" / "dataset")
    ensure_dir(ROOT / "ch2_cnn" / "dataset")
    ensure_dir(ROOT / "ch3_rnn" / "dataset")
    ensure_dir(ROOT / "ch4_lstm" / "dataset")

    tasks = [
        ("breast_cancer", save_breast_cancer_csv),
        ("sms_spam", save_sms_spam_csv),
        ("dry_bean", save_dry_bean_csv),
        ("cnn_datasets", save_cnn_datasets),
        ("reuters", save_reuters_npz),
        ("sine_wave", save_sine_wave_csv),
        ("stock_baseline", save_stock_baseline_csv),
        ("lstm_datasets", save_lstm_datasets),
    ]
    failures = []
    for name, fn in tasks:
        try:
            fn()
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"[error] {name}: {exc}")
    if failures:
        print("[warn] some datasets failed:")
        for name, message in failures:
            print(f" - {name}: {message}")
    print("[done] dataset bootstrap complete")


if __name__ == "__main__":
    main()

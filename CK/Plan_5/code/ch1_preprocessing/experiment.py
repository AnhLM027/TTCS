from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC


DATA_DIR = Path(__file__).resolve().parent / "dataset"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_metrics(name: str, baseline: dict[str, float], processed: dict[str, float]) -> None:
    ensure_results_dir()
    target = RESULTS_DIR / f"{name}_metrics.json"
    target.write_text(
        json.dumps(
            {
                "baseline": baseline,
                "processed": processed,
                "delta_accuracy": processed["accuracy"] - baseline["accuracy"],
                "delta_f1": processed["f1"] - baseline["f1"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def evaluate_classifier(model, x_train, x_test, y_train, y_test) -> dict[str, float]:
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average="weighted")),
    }


def run_titanic() -> None:
    frame = pd.read_csv(DATA_DIR / "titanic.csv")
    y = frame["Survived"]
    x = frame.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric = ["Age", "SibSp", "Parch", "Fare"]
    categorical = ["Pclass", "Sex", "Embarked"]

    baseline = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("num", SimpleImputer(strategy="median"), numeric),
                        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=1_000)),
        ]
    )

    processed = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", RobustScaler())]), numeric),
                        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=1_000, class_weight="balanced")),
        ]
    )

    save_metrics("titanic", evaluate_classifier(baseline, x_train, x_test, y_train, y_test), evaluate_classifier(processed, x_train, x_test, y_train, y_test))


def run_breast_cancer() -> None:
    frame = pd.read_csv(DATA_DIR / "breast_cancer.csv")
    y = frame["target"]
    x = frame.drop(columns=["target"])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline = Pipeline([("model", LogisticRegression(max_iter=2_000))])
    processed = Pipeline(
        [
            ("scale", RobustScaler()),
            ("model", LogisticRegression(max_iter=2_000)),
        ]
    )

    save_metrics(
        "breast_cancer",
        evaluate_classifier(baseline, x_train, x_test, y_train, y_test),
        evaluate_classifier(processed, x_train, x_test, y_train, y_test),
    )


def clean_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"http\S+", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def run_sms_spam() -> None:
    frame = pd.read_csv(DATA_DIR / "sms_spam.csv")
    frame["label"] = frame["label"].map({"ham": 0, "spam": 1})

    x_train, x_test, y_train, y_test = train_test_split(
        frame["message"], frame["label"], test_size=0.2, random_state=42, stratify=frame["label"]
    )

    baseline = Pipeline([("tfidf", TfidfVectorizer(max_features=3_000)), ("model", LogisticRegression(max_iter=1_000))])
    processed = Pipeline(
        [
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, max_features=5_000, ngram_range=(1, 2), stop_words="english")),
            ("model", LogisticRegression(max_iter=1_000, class_weight="balanced")),
        ]
    )

    save_metrics("sms_spam", evaluate_classifier(baseline, x_train, x_test, y_train, y_test), evaluate_classifier(processed, x_train, x_test, y_train, y_test))


def run_dry_bean() -> None:
    frame = pd.read_csv(DATA_DIR / "dry_bean.csv")
    y = frame["Class"]
    x = frame.drop(columns=["Class"])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline = Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf"))])
    processed = Pipeline(
        [("scale", StandardScaler()), ("pca", PCA(n_components=0.95, random_state=42)), ("model", SVC(kernel="rbf"))]
    )

    save_metrics("dry_bean", evaluate_classifier(baseline, x_train, x_test, y_train, y_test), evaluate_classifier(processed, x_train, x_test, y_train, y_test))

    ensure_results_dir()
    reduced = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(StandardScaler().fit_transform(x.sample(min(1_500, len(x)), random_state=42)))
    pd.DataFrame(reduced, columns=["tsne_1", "tsne_2"]).to_csv(RESULTS_DIR / "dry_bean_tsne.csv", index=False)


def main() -> None:
    run_titanic()
    run_breast_cancer()
    run_sms_spam()
    run_dry_bean()
    print(f"[done] results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

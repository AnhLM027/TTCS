from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent


def write_notebook(path: Path, title: str, cells: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb["cells"] = [nbf.v4.new_markdown_cell(f"# {title}")] + cells
    nbf.write(nb, path)
    print(f"[saved] {path}")


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


COMMON_IMPORTS_CH1 = """
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")
"""


def chapter1_notebook(dataset_name: str, load_and_prep: str, model_block: str):
    return [
        md("Notebook này tập trung vào 1 dataset, chạy nhiều mô hình để so sánh và ghi lại kết quả thực nghiệm."),
        code(COMMON_IMPORTS_CH1),
        code(load_and_prep),
        code(model_block),
    ]


COMMON_IMPORTS_DL = """
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")
tf.random.set_seed(42)
np.random.seed(42)
"""


def chapter2_notebook(loader_block: str, num_classes: int, title_suffix: str):
    models_block = f"""
def build_simple_cnn(input_shape, num_classes):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

def build_lenet(input_shape, num_classes):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(6, 5, activation='tanh'),
        layers.AveragePooling2D(),
        layers.Conv2D(16, 5, activation='tanh'),
        layers.AveragePooling2D(),
        layers.Flatten(),
        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),
        layers.Dense(num_classes, activation='softmax')
    ])

def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    return layers.Activation('relu')(x)

def build_resnet_lite(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = residual_block(x, 32)
    x = layers.MaxPooling2D()(x)
    x = residual_block(x, 64)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

def build_vgg_small(input_shape, num_classes):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

builders = {{
    "SimpleCNN": build_simple_cnn,
    "LeNet5": build_lenet,
    "ResNetLite": build_resnet_lite,
    "VGGSmall": build_vgg_small,
}}

results = []
histories = {{}}
for name, builder in builders.items():
    model = builder(input_shape, {num_classes})
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=128, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    results.append({{"model": name, "test_accuracy": acc, "test_loss": loss}})
    histories[name] = history.history

results_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
results_df

ax = sns.barplot(data=results_df, x="test_accuracy", y="model", palette="viridis")
ax.set_title("Model Comparison - {title_suffix}")
plt.show()
"""
    return [
        md("Notebook này chạy 4 kiến trúc CNN trên một dataset ảnh để so sánh hiệu năng."),
        code(COMMON_IMPORTS_DL),
        code(loader_block),
        code(models_block),
    ]


def chapter3_notebook(loader_block: str, task_kind: str):
    if task_kind == "text":
        models_block = """
def build_simple_rnn(vocab_size, num_classes):
    model = keras.Sequential([
        layers.Embedding(vocab_size, 128),
        layers.SimpleRNN(64),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_birnn(vocab_size, num_classes):
    model = keras.Sequential([
        layers.Embedding(vocab_size, 128),
        layers.Bidirectional(layers.SimpleRNN(64)),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_gru(vocab_size, num_classes):
    model = keras.Sequential([
        layers.Embedding(vocab_size, 128),
        layers.GRU(64),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_deep_rnn(vocab_size, num_classes):
    model = keras.Sequential([
        layers.Embedding(vocab_size, 128),
        layers.SimpleRNN(64, return_sequences=True),
        layers.SimpleRNN(64),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

builders = {
    "SimpleRNN": build_simple_rnn,
    "BiRNN": build_birnn,
    "GRU": build_gru,
    "DeepRNN": build_deep_rnn,
}

results = []
for name, builder in builders.items():
    model = builder(vocab_size, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=128, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    results.append({"model": name, "test_accuracy": acc, "test_loss": loss})

results_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
results_df

sns.barplot(data=results_df, x="test_accuracy", y="model", palette="magma")
plt.title("RNN-family Comparison")
plt.show()
"""
    else:
        models_block = """
def build_simple_rnn(input_shape):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.SimpleRNN(64),
        layers.Dense(1)
    ])

def build_birnn(input_shape):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.SimpleRNN(64)),
        layers.Dense(1)
    ])

def build_gru(input_shape):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(64),
        layers.Dense(1)
    ])

def build_deep_rnn(input_shape):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.SimpleRNN(64, return_sequences=True),
        layers.SimpleRNN(64),
        layers.Dense(1)
    ])

builders = {
    "SimpleRNN": build_simple_rnn,
    "BiRNN": build_birnn,
    "GRU": build_gru,
    "DeepRNN": build_deep_rnn,
}

results = []
for name, builder in builders.items():
    model = builder(input_shape)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=64, verbose=0)
    loss, mae = model.evaluate(x_test, y_test, verbose=0)
    results.append({"model": name, "test_mse": loss, "test_mae": mae})

results_df = pd.DataFrame(results).sort_values("test_mse")
results_df

sns.barplot(data=results_df, x="test_mse", y="model", palette="crest")
plt.title("Sequence Forecasting Comparison")
plt.show()
"""
    return [
        md("Notebook này chạy 4 kiến trúc thuộc họ RNN trên một dataset chuỗi để so sánh."),
        code(COMMON_IMPORTS_DL),
        code(loader_block),
        code(models_block),
    ]


def chapter4_notebook(loader_block: str, task_kind: str):
    if task_kind == "forecast":
        models_block = """
def build_lstm(input_shape):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64),
        layers.Dense(1)
    ])

def build_stacked_lstm(input_shape):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(1)
    ])

def build_bilstm(input_shape):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(1)
    ])

def build_attention_lstm(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    attention = layers.Attention()([x, x])
    x = layers.GlobalAveragePooling1D()(attention)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

builders = {
    "StandardLSTM": build_lstm,
    "StackedLSTM": build_stacked_lstm,
    "BiLSTM": build_bilstm,
    "LSTM+Attention": build_attention_lstm,
}

results = []
for name, builder in builders.items():
    model = builder(input_shape)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=64, verbose=0)
    loss, mae = model.evaluate(x_test, y_test, verbose=0)
    results.append({"model": name, "test_mse": loss, "test_mae": mae})

results_df = pd.DataFrame(results).sort_values("test_mse")
results_df

sns.barplot(data=results_df, x="test_mse", y="model", palette="rocket")
plt.title("LSTM-family Comparison")
plt.show()
"""
    else:
        models_block = """
text = text[:200000]
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
seq_len = 80
step = 3
sentences, next_chars = [], []
for i in range(0, len(text) - seq_len, step):
    sentences.append(text[i:i+seq_len])
    next_chars.append(text[i+seq_len])

x = np.zeros((len(sentences), seq_len, len(chars)), dtype=np.float32)
y = np.zeros((len(sentences), len(chars)), dtype=np.float32)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_idx[char]] = 1.0
    y[i, char_to_idx[next_chars[i]]] = 1.0

x_train, x_test = x[:20000], x[20000:24000]
y_train, y_test = y[:20000], y[20000:24000]

def build_lstm(input_shape):
    return keras.Sequential([layers.Input(shape=input_shape), layers.LSTM(128), layers.Dense(len(chars), activation='softmax')])

def build_stacked_lstm(input_shape):
    return keras.Sequential([layers.Input(shape=input_shape), layers.LSTM(128, return_sequences=True), layers.LSTM(128), layers.Dense(len(chars), activation='softmax')])

def build_bilstm(input_shape):
    return keras.Sequential([layers.Input(shape=input_shape), layers.Bidirectional(layers.LSTM(128)), layers.Dense(len(chars), activation='softmax')])

def build_attention_lstm(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Attention()([x, x])
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(len(chars), activation='softmax')(x)
    return keras.Model(inputs, outputs)

builders = {
    "StandardLSTM": build_lstm,
    "StackedLSTM": build_stacked_lstm,
    "BiLSTM": build_bilstm,
    "LSTM+Attention": build_attention_lstm,
}

results = []
for name, builder in builders.items():
    model = builder((seq_len, len(chars)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    results.append({"model": name, "test_accuracy": acc, "test_loss": loss})

results_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
results_df

sns.barplot(data=results_df, x="test_accuracy", y="model", palette="flare")
plt.title("Text Generation Next-Char Comparison")
plt.show()
"""
    return [
        md("Notebook này chạy các biến thể LSTM trên đúng một dataset để so sánh kết quả thực nghiệm."),
        code(COMMON_IMPORTS_DL),
        code(loader_block),
        code(models_block),
    ]


def generate_ch1():
    base = ROOT / "ch1_preprocessing" / "notebooks"
    write_notebook(
        base / "titanic_experiment.ipynb",
        "Titanic - Preprocessing And Model Comparison",
        chapter1_notebook(
            "titanic",
            """
df = pd.read_csv(DATA_DIR / "titanic.csv")
display(df.head())
display(df.isna().mean().sort_values(ascending=False).head(10))

target = "Survived"
feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[feature_cols]
y = df[target]

numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            """,
            """
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVC": SVC(probability=True, random_state=42),
}

results = []
for name, estimator in models.items():
    pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
    })

results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
display(results_df)
sns.barplot(data=results_df, x="accuracy", y="model", palette="Blues_r")
plt.title("Titanic Model Comparison")
plt.show()
            """,
        ),
    )

    write_notebook(
        base / "breast_cancer_experiment.ipynb",
        "Breast Cancer - Scaling And Classifier Comparison",
        chapter1_notebook(
            "breast_cancer",
            """
df = pd.read_csv(DATA_DIR / "breast_cancer.csv")
display(df.head())

X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            """,
            """
models = {
    "LogisticRegression": Pipeline([("scaler", RobustScaler()), ("model", LogisticRegression(max_iter=2000))]),
    "SVC": Pipeline([("scaler", StandardScaler()), ("model", SVC())]),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
    })

results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
display(results_df)
sns.barplot(data=results_df, x="accuracy", y="model", palette="Greens_r")
plt.title("Breast Cancer Model Comparison")
plt.show()
            """,
        ),
    )

    write_notebook(
        base / "sms_spam_experiment.ipynb",
        "SMS Spam - Text Cleaning And Model Comparison",
        chapter1_notebook(
            "sms_spam",
            """
df = pd.read_csv(DATA_DIR / "sms_spam.csv")
df["label"] = df["label"].map({"ham": 0, "spam": 1})
display(df.head())

X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)
            """,
            """
vectorizers = {
    "tfidf_unigram": TfidfVectorizer(max_features=5000, stop_words="english"),
    "tfidf_bigram": TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english"),
}

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVCProxy": SVC(kernel="linear"),
}

results = []
for vec_name, vectorizer in vectorizers.items():
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    for model_name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        results.append({
            "setup": f"{vec_name}+{model_name}",
            "accuracy": accuracy_score(y_test, preds),
            "f1_weighted": f1_score(y_test, preds, average="weighted"),
        })

results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
display(results_df)
sns.barplot(data=results_df, x="accuracy", y="setup", palette="Oranges_r")
plt.title("SMS Spam Comparison")
plt.show()
            """,
        ),
    )

    write_notebook(
        base / "dry_bean_experiment.ipynb",
        "Dry Bean - PCA And Model Comparison",
        chapter1_notebook(
            "dry_bean",
            """
df = pd.read_excel(DATA_DIR / "dry_bean.xlsx")
display(df.head())

X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            """,
            """
scaled_train = StandardScaler().fit_transform(X_train)
scaled_test = StandardScaler().fit_transform(X_test)

models = {
    "SVC_raw": SVC(kernel="rbf"),
    "LogReg_PCA95": Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=0.95, random_state=42)), ("model", LogisticRegression(max_iter=2000))]),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
}

results = []
for name, model in models.items():
    if name == "SVC_raw":
        model.fit(scaled_train, y_train)
        preds = model.predict(scaled_test)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
    })

results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
display(results_df)
sns.barplot(data=results_df, x="accuracy", y="model", palette="Purples_r")
plt.title("Dry Bean Model Comparison")
plt.show()
            """,
        ),
    )


def generate_ch2():
    base = ROOT / "ch2_cnn" / "notebooks"
    load_blocks = {
        "mnist_experiment.ipynb": (
            "MNIST - CNN Model Comparison",
            """
data = np.load(DATA_DIR / "mnist.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]
x_train = np.expand_dims(x_train / 255.0, -1)
x_test = np.expand_dims(x_test / 255.0, -1)
input_shape = x_train.shape[1:]
            """,
            10,
            "MNIST",
        ),
        "fashion_mnist_experiment.ipynb": (
            "Fashion-MNIST - CNN Model Comparison",
            """
data = np.load(DATA_DIR / "fashion_mnist.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]
x_train = np.expand_dims(x_train / 255.0, -1)
x_test = np.expand_dims(x_test / 255.0, -1)
input_shape = x_train.shape[1:]
            """,
            10,
            "Fashion-MNIST",
        ),
        "cifar10_experiment.ipynb": (
            "CIFAR-10 - CNN Model Comparison",
            """
data = np.load(DATA_DIR / "cifar10.npz")
x_train, y_train = data["x_train"], data["y_train"].reshape(-1)
x_test, y_test = data["x_test"], data["y_test"].reshape(-1)
x_train = x_train / 255.0
x_test = x_test / 255.0
input_shape = x_train.shape[1:]
            """,
            10,
            "CIFAR-10",
        ),
        "pneumoniamnist_experiment.ipynb": (
            "PneumoniaMNIST - CNN Model Comparison",
            """
data = np.load(DATA_DIR / "pneumoniamnist.npz")
x_train, y_train = data["train_images"], data["train_labels"].reshape(-1)
x_test, y_test = data["test_images"], data["test_labels"].reshape(-1)
x_train = np.expand_dims(x_train / 255.0, -1)
x_test = np.expand_dims(x_test / 255.0, -1)
input_shape = x_train.shape[1:]
            """,
            2,
            "PneumoniaMNIST",
        ),
    }
    for name, (title, block, n_classes, suffix) in load_blocks.items():
        write_notebook(base / name, title, chapter2_notebook(block, n_classes, suffix))


def generate_ch3():
    base = ROOT / "ch3_rnn" / "notebooks"
    write_notebook(
        base / "imdb_experiment.ipynb",
        "IMDB Sentiment - RNN Family Comparison",
        chapter3_notebook(
            """
df = pd.read_csv(DATA_DIR / "IMDB Dataset.csv").sample(10000, random_state=42)
texts = df["review"].to_numpy()
labels = df["sentiment"].map({"negative": 0, "positive": 1}).to_numpy()

tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<unk>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=200)
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
vocab_size = 10000
num_classes = 2
            """,
            "text",
        ),
    )
    write_notebook(
        base / "reuters_experiment.ipynb",
        "Reuters Newswire - RNN Family Comparison",
        chapter3_notebook(
            """
data = np.load(DATA_DIR / "reuters.npz", allow_pickle=True)
x_train = keras.preprocessing.sequence.pad_sequences(data["x_train"], maxlen=200)
x_test = keras.preprocessing.sequence.pad_sequences(data["x_test"], maxlen=200)
y_train = data["y_train"]
y_test = data["y_test"]
vocab_size = 10000
num_classes = len(np.unique(y_train))
            """,
            "text",
        ),
    )
    write_notebook(
        base / "sine_wave_experiment.ipynb",
        "Sine Wave - Sequence Forecasting Comparison",
        chapter3_notebook(
            """
df = pd.read_csv(DATA_DIR / "sine_wave.csv")
series = df["signal"].to_numpy().astype("float32")
window = 40
X, y = [], []
for i in range(len(series) - window):
    X.append(series[i:i+window])
    y.append(series[i+window])
X = np.array(X)[..., None]
y = np.array(y)
split = int(0.8 * len(X))
x_train, x_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
input_shape = x_train.shape[1:]
            """,
            "forecast",
        ),
    )
    write_notebook(
        base / "stock_baseline_experiment.ipynb",
        "Stock Baseline - Sequence Forecasting Comparison",
        chapter3_notebook(
            """
df = pd.read_csv(DATA_DIR / "aapl.csv")
close_col = [c for c in df.columns if "AAPL.Close" in c or c.lower() == "close"][0]
series = df[close_col].to_numpy().astype("float32")
series = (series - series.min()) / (series.max() - series.min())
window = 30
X, y = [], []
for i in range(len(series) - window):
    X.append(series[i:i+window])
    y.append(series[i+window])
X = np.array(X)[..., None]
y = np.array(y)
split = int(0.8 * len(X))
x_train, x_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
input_shape = x_train.shape[1:]
            """,
            "forecast",
        ),
    )


def generate_ch4():
    base = ROOT / "ch4_lstm" / "notebooks"
    forecast_blocks = {
        "google_stock_experiment.ipynb": (
            "Google Stock - LSTM Family Comparison",
            """
df = pd.read_csv(DATA_DIR / "google_stock.csv")
close_col = [c for c in df.columns if c.lower() == "close" or "close" in c.lower()][0]
series = df[close_col].to_numpy().astype("float32")
series = (series - series.min()) / (series.max() - series.min())
window = 30
X, y = [], []
for i in range(len(series) - window):
    X.append(series[i:i+window])
    y.append(series[i+window])
X = np.array(X)[..., None]
y = np.array(y)
split = int(0.8 * len(X))
x_train, x_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
input_shape = x_train.shape[1:]
            """,
        ),
        "weather_experiment.ipynb": (
            "Weather Forecasting - LSTM Family Comparison",
            """
import zipfile
with zipfile.ZipFile(DATA_DIR / "jena_climate.zip") as archive:
    csv_name = [name for name in archive.namelist() if name.endswith(".csv")][0]
    with archive.open(csv_name) as handle:
        df = pd.read_csv(handle)
features = df[["T (degC)", "p (mbar)", "rho (g/m**3)"]].astype("float32")
features = (features - features.min()) / (features.max() - features.min())
values = features.to_numpy()
window = 72
X, y = [], []
for i in range(len(values) - window):
    X.append(values[i:i+window])
    y.append(values[i+window, 0])
X = np.array(X)
y = np.array(y)
split = int(0.8 * len(X))
x_train, x_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
input_shape = x_train.shape[1:]
            """,
        ),
        "air_quality_experiment.ipynb": (
            "Air Quality - LSTM Family Comparison",
            """
import zipfile
with zipfile.ZipFile(DATA_DIR / "air_quality.zip") as archive:
    csv_name = [name for name in archive.namelist() if name.endswith(".csv")][0]
    with archive.open(csv_name) as handle:
        df = pd.read_csv(handle, sep=';', decimal=',')
df = df.replace(-200, np.nan).dropna()
feature_cols = [col for col in df.columns if 'CO(' in col or 'NOx' in col or col == 'T']
features = df[feature_cols].astype("float32")
features = (features - features.min()) / (features.max() - features.min())
values = features.to_numpy()
window = 48
X, y = [], []
for i in range(len(values) - window):
    X.append(values[i:i+window])
    y.append(values[i+window, 0])
X = np.array(X)
y = np.array(y)
split = int(0.8 * len(X))
x_train, x_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
input_shape = x_train.shape[1:]
            """,
        ),
    }
    for name, (title, block) in forecast_blocks.items():
        write_notebook(base / name, title, chapter4_notebook(block, "forecast"))

    write_notebook(
        base / "text_generation_experiment.ipynb",
        "Text Generation - LSTM Family Comparison",
        chapter4_notebook(
            """
text = (DATA_DIR / "tiny_shakespeare.txt").read_text(encoding="utf-8")
print(text[:1000])
            """,
            "text_generation",
        ),
    )


def main():
    generate_ch1()
    generate_ch2()
    generate_ch3()
    generate_ch4()
    print("[done] Plan_5 notebooks generated")


if __name__ == "__main__":
    main()

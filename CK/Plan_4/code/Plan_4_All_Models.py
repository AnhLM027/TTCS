import os

# Cấu hình TensorFlow chạy trên CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
print("Đã cấu hình TensorFlow chạy trên CPU.")

import pandas as pd
import numpy as np
import re
import string
import time
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, SimpleRNN, LSTM

MAX_WORDS = 10000  # Giới hạn số lượng từ vựng
MAX_LEN = 200      # Giới hạn chiều dài mỗi đoạn text

# ==============================================================================
# CHƯƠNG 1: TIỀN XỬ LÝ DỮ LIỆU & EDA
# ==============================================================================
print('\n' + '='*50)
print('CHƯƠNG 1: TIỀN XỬ LÝ DỮ LIỆU & EDA')
print('='*50)

# 1. Load data
print('Đọc dữ liệu từ file CSV...')
path = '../../../dataset/IMDB Dataset.csv' if not os.path.exists('dataset/IMDB Dataset.csv') else 'dataset/IMDB Dataset.csv'
df = pd.read_csv(path)

# BỔ SUNG: Tính độ dài trước khi xử lý
df['len_before'] = df['review'].apply(len)

print(f'Số lượng mẫu ban đầu: {len(df)}')
print(df.head())

# 2. Text Cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text) # Bỏ nội dung trong ngoặc vuông
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Bỏ URL
    text = re.sub(r'<.*?>+', '', text) # Bỏ thẻ HTML
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Bỏ dấu câu
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text) # Bỏ chữ chứa số
    return text

print('\nĐang làm sạch văn bản...')
df['review_cleaned'] = df['review'].apply(clean_text)
df['len_after'] = df['review_cleaned'].apply(len)

# BỔ SUNG: Visualize so sánh EDA
print('Đang tạo biểu đồ EDA...')
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['len_before'], bins=50, color='skyblue', kde=True)
plt.title('Độ dài văn bản (Trước khi xử lý)')
plt.xlabel('Số ký tự')

plt.subplot(1, 2, 2)
sns.histplot(df['len_after'], bins=50, color='salmon', kde=True)
plt.title('Độ dài văn bản (Sau khi xử lý)')
plt.xlabel('Số ký tự')
plt.savefig('text_length_comparison.png')

# BỔ SUNG 3: Phân bố tích lũy chiều dài văn bản
plt.figure(figsize=(8, 5))
sns.ecdfplot(df['len_after'], color='purple')
plt.axvline(x=200, color='red', linestyle='--', label='MAX_LEN = 200')
plt.title('Phân bố tích lũy chiều dài văn bản')
plt.xlabel('Độ dài (ký tự)')
plt.ylabel('Tỷ lệ phần trăm dữ liệu')
plt.legend()
plt.savefig('cumulative_length_distribution.png')

# 3. Encode Labels (positive -> 1, negative -> 0)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# BỔ SUNG 1: Biểu đồ phân phối nhãn (Label Distribution)
print('Đang tạo biểu đồ phân phối nhãn...')
plt.figure(figsize=(6, 5))
sns.countplot(x='sentiment', hue='sentiment', data=df, palette='viridis', legend=False)
plt.title('Phân phối nhãn (Class Distribution)')
plt.xticks([0, 1], ['Negative (0)', 'Positive (1)'])
plt.savefig('label_distribution.png')

# BỔ SUNG 2: WordCloud theo từng class
print('Đang tạo WordCloud cho từng class...')
pos_words = ' '.join(df[df['sentiment'] == 1]['review_cleaned'])
neg_words = ' '.join(df[df['sentiment'] == 0]['review_cleaned'])

wc_pos = WordCloud(width=400, height=400, background_color='white', colormap='Greens', max_words=100).generate(pos_words)
wc_neg = WordCloud(width=400, height=400, background_color='white', colormap='Reds', max_words=100).generate(neg_words)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud - Positive')

plt.subplot(1, 2, 2)
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud - Negative')
plt.savefig('wordcloud_by_class.png')
texts = df['review_cleaned'].values
labels = df['sentiment'].values

# 4. Tokenization (Chuyển text thành số)
print('Đang Tokenize văn bản...')
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# 5. Padding (Cắt / Thêm số 0 để các mảng dài bằng nhau)
data = pad_sequences(sequences, maxlen=MAX_LEN)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print('\nHoàn tất tiền xử lý!')
print(f'Train size: {X_train.shape}, Test size: {X_test.shape}')


# ==============================================================================
# CHƯƠNG 2: XÂY DỰNG VÀ HUẤN LUYỆN CNN
# ==============================================================================
print('\n' + '='*50)
print('CHƯƠNG 2: CNN')
print('='*50)

def build_cnn():
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn()
cnn_model.summary()
# BỔ SUNG 5: Sơ đồ kiến trúc Model
try:
    plot_model(cnn_model, to_file='model_cnn_architecture.png', show_shapes=True, show_layer_names=True)
except Exception as e:
    print("Không thể vẽ sơ đồ kiến trúc CNN (cần cài đặt pydot và graphviz).")

print('Bắt đầu huấn luyện CNN...')
start_time = time.time()
cnn_history = cnn_model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)
cnn_time = time.time() - start_time


# ==============================================================================
# CHƯƠNG 3: XÂY DỰNG VÀ HUẤN LUYỆN RNN
# ==============================================================================
print('\n' + '='*50)
print('CHƯƠNG 3: RNN')
print('='*50)

def build_rnn():
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        SimpleRNN(64),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

rnn_model = build_rnn()
rnn_model.summary()
try:
    plot_model(rnn_model, to_file='model_rnn_architecture.png', show_shapes=True, show_layer_names=True)
except Exception as e:
    print("Không thể vẽ sơ đồ kiến trúc RNN (cần cài đặt pydot và graphviz).")

print('Bắt đầu huấn luyện RNN...')
start_time = time.time()
rnn_history = rnn_model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)
rnn_time = time.time() - start_time


# ==============================================================================
# CHƯƠNG 4: XÂY DỰNG VÀ HUẤN LUYỆN LSTM
# ==============================================================================
print('\n' + '='*50)
print('CHƯƠNG 4: LSTM')
print('='*50)

def build_lstm():
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

lstm_model = build_lstm()
lstm_model.summary()
try:
    plot_model(lstm_model, to_file='model_lstm_architecture.png', show_shapes=True, show_layer_names=True)
except Exception as e:
    print("Không thể vẽ sơ đồ kiến trúc LSTM (cần cài đặt pydot và graphviz).")

print('Bắt đầu huấn luyện LSTM...')
start_time = time.time()
lstm_history = lstm_model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)
lstm_time = time.time() - start_time


# ==============================================================================
# PHẦN KẾT: ĐÁNH GIÁ VÀ SO SÁNH
# ==============================================================================
print('\n' + '='*50)
print('ĐÁNH GIÁ VÀ SO SÁNH HIỆU NĂNG')
print('='*50)

print('Đánh giá CNN...')
cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)

print('Đánh giá RNN...')
rnn_loss, rnn_acc = rnn_model.evaluate(X_test, y_test, verbose=0)

print('Đánh giá LSTM...')
lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test, verbose=0)

models = ['CNN', 'RNN', 'LSTM']
accuracies = [cnn_acc, rnn_acc, lstm_acc]

print(f'\nKết quả Accuracy trên tập Test:\nCNN:  {cnn_acc:.4f}\nRNN:  {rnn_acc:.4f}\nLSTM: {lstm_acc:.4f}\n')

# 1. Biểu đồ cột so sánh Accuracy
plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies, palette='magma')
plt.title('So sánh độ chính xác (Accuracy) trên tập Test')
plt.ylabel('Độ chính xác')
plt.ylim([0, 1.0])
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, str(round(v, 4)), ha='center', fontweight='bold')
plt.savefig('accuracy_comparison.png')

# BỔ SUNG 4: Learning Curve đầy đủ (Train vs Validation)
histories = [cnn_history, rnn_history, lstm_history]
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
for i, (model_name, history) in enumerate(zip(models, histories)):
    # Accuracy
    axes[i, 0].plot(history.history['accuracy'], label='Train Acc')
    axes[i, 0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[i, 0].set_title(f'{model_name} Accuracy')
    axes[i, 0].set_xlabel('Epochs')
    axes[i, 0].set_ylabel('Accuracy')
    axes[i, 0].legend()
    # Loss
    axes[i, 1].plot(history.history['loss'], label='Train Loss')
    axes[i, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[i, 1].set_title(f'{model_name} Loss')
    axes[i, 1].set_xlabel('Epochs')
    axes[i, 1].set_ylabel('Loss')
    axes[i, 1].legend()
plt.tight_layout()
plt.savefig('learning_curves_full.png')

# BỔ SUNG 6: Training time comparison
times = [cnn_time, rnn_time, lstm_time]
plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=times, palette='viridis')
plt.title('So sánh thời gian huấn luyện (Training Time)')
plt.ylabel('Thời gian (giây)')
for i, v in enumerate(times):
    plt.text(i, v + max(times)*0.02, f'{v:.2f}s', ha='center', fontweight='bold')
plt.savefig('training_time_comparison.png')

# 3. Classification Reports
print("\nĐang thực hiện dự đoán để in Classification Reports...")
y_pred_cnn = (cnn_model.predict(X_test, verbose=0) > 0.5).astype(int)
y_pred_rnn = (rnn_model.predict(X_test, verbose=0) > 0.5).astype(int)
y_pred_lstm = (lstm_model.predict(X_test, verbose=0) > 0.5).astype(int)

# BỔ SUNG: Biểu đồ Precision / Recall / F1
precisions = []
recalls = []
f1_scores = []
for pred in [y_pred_cnn, y_pred_rnn, y_pred_lstm]:
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, average='binary')
    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f)

metrics_df = pd.DataFrame({
    'Model': models * 3,
    'Metric': ['Precision']*3 + ['Recall']*3 + ['F1-Score']*3,
    'Score': precisions + recalls + f1_scores
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df, palette='muted')
plt.title('So sánh Precision, Recall, F1-Score')
plt.ylim([0, 1.0])
plt.legend(loc='lower right')
plt.savefig('metrics_comparison.png')

print("\n--- Classification Report CNN ---")
print(classification_report(y_test, y_pred_cnn, target_names=['Negative', 'Positive']))
print("\n--- Classification Report RNN ---")
print(classification_report(y_test, y_pred_rnn, target_names=['Negative', 'Positive']))
print("\n--- Classification Report LSTM ---")
print(classification_report(y_test, y_pred_lstm, target_names=['Negative', 'Positive']))

# 4. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (m, name, pred) in enumerate(zip([cnn_model, rnn_model, lstm_model], models, [y_pred_cnn, y_pred_rnn, y_pred_lstm])):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    axes[i].set_title(f'CM: {name}')
plt.tight_layout()
plt.savefig('confusion_matrices.png')

# 5. ROC Curves
plt.figure(figsize=(8, 6))
for m, name in zip([cnn_model, rnn_model, lstm_model], models):
    y_prob = m.predict(X_test, verbose=0)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.3f})')
plt.plot([0, 1], [0, 1], 'k--'); plt.legend(); plt.title('ROC Curves Comparison')
plt.savefig('roc_curves.png')

print("\nHOÀN TẤT!")

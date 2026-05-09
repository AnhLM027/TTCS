

# FILE: code/ch1_preprocessing/ch1.ipynb


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

# Cấu hình thẩm mỹ cho biểu đồ
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. Tải dataset gốc (Kaggle version)
print("--- 1. Loading Dataset ---")
df = pd.read_csv('./dataset/titanic.csv')
print(df.head())

```


```python
# 2. Phân tích và Trực quan hóa dữ liệu (Exploratory Data Analysis - EDA)
print("\n--- 2. Visualizing Data ---")

# Tạo một lưới biểu đồ 3x2
fig, axes = plt.subplots(3, 2, figsize=(16, 20))

# 1. Tỉ lệ sống sót tổng quan (Pie chart)
df['Survived'].value_counts().plot.pie(
    explode=[0, 0.1], autopct='%1.1f%%', ax=axes[0, 0], shadow=True, colors=['#ff9999','#66b3ff']
)
axes[0, 0].set_title('Overall Survival Rate')
axes[0, 0].set_ylabel('')

# 2. Sống sót theo Giới tính (Countplot)
sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[0, 1], palette='magma')
axes[0, 1].set_title('Survival by Sex')

# 3. Sống sót theo Hạng vé (Pclass)
sns.barplot(data=df, x='Pclass', y='Survived', ax=axes[1, 0], palette='viridis')
axes[1, 0].set_title('Survival Rate by Class')

# 4. Phân phối Tuổi (Distribution Plot)
sns.histplot(df[df['Survived'] == 1]['Age'].dropna(), color='green', label='Survived', kde=True, ax=axes[1, 1], alpha=0.5)
sns.histplot(df[df['Survived'] == 0]['Age'].dropna(), color='red', label='Not Survived', kde=True, ax=axes[1, 1], alpha=0.5)
axes[1, 1].set_title('Age Distribution by Survival')
axes[1, 1].legend()

# 5. Mối quan hệ giữa Giá vé và Hạng vé (Boxplot)
sns.boxplot(data=df, x='Pclass', y='Fare', hue='Survived', ax=axes[2, 0], palette='Set2')
axes[2, 0].set_title('Fare Distribution by Pclass & Survival')
axes[2, 0].set_ylim(0, 300) # Giới hạn để dễ quan sát

# 6. Sống sót theo Cổng lên tàu (Embarked)
sns.pointplot(data=df, x='Embarked', y='Survived', hue='Sex', ax=axes[2, 1], palette='coolwarm')
axes[2, 1].set_title('Survival Probability by Embarkation & Sex')

plt.tight_layout()
plt.show()

# Biểu đồ Heatmap tương quan giữa các biến số
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='RdBu_r', center=0, fmt='.2f')
plt.title('Correlation Heatmap of Numeric Features')
plt.show()

```


```python
# 3. Tiền xử lý dữ liệu (Preprocessing)
print("\n--- 3. Data Preprocessing ---")

# 3.1 Xử lý giá trị khuyết
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# 3.2 Trích xuất Title từ Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# 3.3 Kỹ nghệ đặc trưng
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Visualize Title vs Survival
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='Title', y='Survived', palette='pastel')
plt.title('Survival Rate by Title')
plt.show()

# 3.4 Loại bỏ các cột thừa
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 3.5 Mã hóa
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Title'] = le.fit_transform(df['Title'])
df = pd.get_dummies(df, columns=['Embarked'])

print("\nData after preprocessing:")
print(df.head())

```


```python
# 4. Chia dữ liệu Train/Test
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

```


```python
# 5. Build & Train Model (Logistic Regression)
print("\n--- 4. Building & Training Model ---")
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

```


```python
# 6. Đánh giá (Evaluation)
y_pred = model.predict(X_test_scaled)
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Titanic')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

```



# FILE: code/ch2_cnn/ch2.ipynb


```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.metrics import confusion_matrix
import seaborn as sns

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.utils.set_random_seed(SEED)

# 1. Tải và chuẩn bị Dataset Fashion MNIST
print("--- 1. Loading Fashion MNIST Dataset ---")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Tạo biến size riêng cho height và width
img_height, img_width = 64, 64

# Normalize
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# thêm channel dimension
train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

# resize thật sự
train_images = tf.image.resize(train_images, (img_height, img_width)).numpy()
test_images = tf.image.resize(test_images, (img_height, img_width)).numpy()

print(train_images.shape)   # (60000, 64, 64, 1)
print(test_images.shape)    # (10000, 64, 64, 1)

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

```


```python
# 2. Visualize một số hình ảnh mẫu
print("\n--- 2. Visualizing Samples ---")
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].reshape(img_height, img_width), cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

```


```python
# 3. Xây dựng mô hình CNN cải tiến (Improved Architecture)
print("\n--- 3. Building CNN Model ---")

model = models.Sequential([
    # Layer 1: Convolution + BatchNorm + Pooling + Dropout
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Layer 2: Convolution + BatchNorm + Pooling + Dropout
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Layer 3: Convolution + BatchNorm + Flatten
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    
    # Fully Connected Layers
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

```


```python
# 4. Huấn luyện (Training) với Early Stopping
print("\n--- 4. Training Model ---")

# Early stopping để tránh overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(train_images, train_labels, epochs=15, 
                    validation_split=0.2, 
                    callbacks=[early_stop],
                    verbose=1)

```


```python
# 5. Đánh giá trên tập Test (Evaluation)
print("\n--- 5. Evaluating Model ---")
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'\nTest Accuracy: {test_acc:.4f}')

```


```python
# 6. Visualize kết quả huấn luyện
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

```


```python
# 7. Dự đoán và hiển thị Confusion Matrix
print("\n--- 7. Confusion Matrix ---")
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(test_labels, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Fashion MNIST')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```


```python
# 8. Hiển thị dự đoán trên các mẫu ngẫu nhiên
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img.reshape(img_height, img_width), cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel(f"{class_names[predicted_label]} {100*np.max(predictions_array):2.0f}% ({class_names[true_label]})", color=color)

plt.figure(figsize=(15, 10))
num_rows, num_cols = 5, 3
for i in range(num_rows * num_cols):
    idx = np.random.randint(0, test_images.shape[0])
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(idx, y_pred[idx], test_labels, test_images)
plt.tight_layout()
plt.show()
```



# FILE: code/ch3_rnn/ch3.ipynb


```python
import os
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# FIX RANDOM SEED (để kết quả chạy ổn định)
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Nếu dùng TF >= 2.9
tf.keras.utils.set_random_seed(SEED)

print(f"Đã gắn seed = {SEED}")

# 1. Tải và chuẩn bị Dataset từ CSV
print("--- 1. Loading Local CSV Dataset ---")

max_features = 10000
maxlen = 500
csv_path = 'dataset/IMDB Dataset.csv'

df = pd.read_csv(csv_path)

print(f"Tổng số bản ghi: {len(df)}")

df.head(10)
```


```python
# Chuyển nhãn về số
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) # Remove HTML
    text = re.sub(r'[^a-z\s]', '', text) # Remove special chars
    return text

df['review'] = df['review'].apply(clean_text)

df.head(10)
```


```python
# 1.1.1 Xử lý Stop words và Tokenization
import nltk
from nltk.corpus import stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stop_words.update(['br', 'movie', 'film'])

print("Đang lọc stop words...")
def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w not in stop_words])

df['review'] = df['review'].apply(remove_stopwords)

print("Đang Tokenizing...")
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['review'])

X = tokenizer.texts_to_sequences(df['review'])
y = df['sentiment'].values

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train sequences: {len(X_train)}")
print(f"Test sequences: {len(X_test)}")

# Cập nhật review_lengths cho các cell sau
review_lengths = [len(x) for x in X_train]

# Để decode được, ta cần reverse_word_index
word_index = tokenizer.word_index
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
```


```python
# 1.2 Khám phá dữ liệu (Data Exploration)
import numpy as np

# Giải mã một bài đánh giá mẫu
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review if i != 0])

print("Review mẫu:")
print(decode_review(X_train[0]))
print("\nNhãn (1: Tích cực, 0: Tiêu cực):", y_train[0])

# Thống kê độ dài các bài đánh giá
review_lengths = [len(x) for x in X_train]
print(f"\nĐộ dài trung bình: {np.mean(review_lengths):.2f}")
print(f"Độ dài lớn nhất: {np.max(review_lengths)}")
print(f"Độ dài nhỏ nhất: {np.min(review_lengths)}")
```


```python
import seaborn as sns
# Visualize phân phối độ dài bài đánh giá
plt.figure(figsize=(10, 5))
sns.histplot(review_lengths, bins=50, kde=True, color='blue')
plt.axvline(maxlen, color='red', linestyle='--', label=f'Maxlen = {maxlen}')
plt.title('Phân phối độ dài các bài đánh giá (Review Length Distribution)')
plt.xlabel('Số lượng từ')
plt.ylabel('Số lượng bài đánh giá')
plt.legend()
plt.show()
```


```python
# 1.2 Phân phối nhãn (Sentiment Distribution)
import pandas as pd
unique, counts = np.unique(y_train, return_counts=True)
plt.figure(figsize=(8, 5))
sns.barplot(x=['Negative (0)', 'Positive (1)'], y=counts, palette='viridis', hue=['Negative (0)', 'Positive (1)'], legend=False)
plt.title('Phân phối nhãn trong tập huấn luyện')
plt.ylabel('Số lượng bài đánh giá')
plt.show()
```


```python
# 1.3 So sánh độ dài bài đánh giá giữa hai nhãn
df_lengths = pd.DataFrame({'length': review_lengths, 'sentiment': y_train})
plt.figure(figsize=(10, 6))
sns.boxplot(x='sentiment', y='length', data=df_lengths, palette='Set2', hue='sentiment', legend=False)
plt.title('So sánh độ dài bài đánh giá theo nhãn')
plt.xlabel('Nhãn (0: Tiêu cực, 1: Tích cực)')
plt.ylabel('Số lượng từ')
plt.ylim(0, 1000) # Giới hạn 1000 từ để dễ quan sát phần chính
plt.show()
```


```python
# 1.4 Thống kê các từ xuất hiện nhiều nhất
from collections import Counter
all_words_indices = []
for review in X_train:
    all_words_indices.extend(review)

# Giải mã các chỉ số thành từ (loại bỏ các token đặc biệt 0, 1, 2)
top_words = Counter([reverse_word_index.get(i - 3, '?') for i in all_words_indices if i > 3]).most_common(20)
words, counts = zip(*top_words)

plt.figure(figsize=(12, 6))
sns.barplot(x=list(counts), y=list(words), palette='magma', hue=list(words), legend=False)
plt.title('Top 20 từ xuất hiện nhiều nhất trong Dataset')
plt.xlabel('Số lần xuất hiện')
plt.show()

# 1.5 Word Cloud
try:
    from wordcloud import WordCloud
    word_freq = {word: count for word, count in top_words}
    wc = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate_from_frequencies(word_freq)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - Các từ phổ biến nhất')
    plt.show()
except ImportError:
    print("Thư viện 'wordcloud' chưa được cài đặt. Bạn có thể cài đặt bằng lệnh: pip install wordcloud")
```


```python
# 2. Tiền xử lý (Padding)
print("\n--- 2. Preprocessing: Padding Sequences ---")
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

```


```python
# 3. Xây dựng mô hình SimpleRNN (Chống Overfitting)
print("\n--- 3. Building SimpleRNN Model ---")

model = models.Sequential([
    layers.Embedding(max_features, 32, input_length=maxlen),
    # Giữ nguyên SimpleRNN theo yêu cầu chương 3
    # Thêm dropout và recurrent_dropout để hạn chế học vẹt
    layers.SimpleRNN(32, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

# Sử dụng optimizer 'adam' với learning rate mặc định ổn định hơn
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

```


```python
# 4. Huấn luyện (Training) với Early Stopping
import tensorflow as tf # Đảm bảo 'tf' luôn được định nghĩa
print("\n--- 4. Training Model ---")

# Dừng sớm ngay khi val_loss có dấu hiệu tăng (patience=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=15, 
                    batch_size=64, validation_split=0.2,
                    callbacks=[early_stop])

```


```python
# 5. Đánh giá (Evaluation)
print("\n--- 5. Evaluating Model ---")
results = model.evaluate(X_test, y_test)
print(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}")

```


```python
# 6. Visualize kết quả huấn luyện
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('RNN Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('RNN Loss')
plt.legend()
plt.show()

```


```python
# 7. Dự đoán trên dữ liệu thực tế (Custom Inference)
print("\n--- 7. Testing with custom reviews ---")

def predict_sentiment(text):
    # Tokenize và tiền xử lý text nhập vào bằng tokenizer cục bộ
    review = tokenizer.texts_to_sequences([text])
    review = pad_sequences(review, maxlen=maxlen)
    
    prediction = model.predict(review)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    print(f"Review: '{text}'")
    print(f"Predicted Sentiment: {sentiment} ({prediction[0][0]:.4f})\n")

predict_sentiment("This movie was absolutely amazing and wonderful")
predict_sentiment("It was a waste of time and very boring")

```


```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# 8. Báo cáo phân loại chi tiết
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - IMDB Sentiment')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```


```python
# 9. Đường cong ROC (ROC Curve)
from sklearn.metrics import roc_curve, auc

y_pred_probs = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
```



# FILE: code/ch4_lstm/ch4.ipynb


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Tải Dataset (Airline Passengers)
print("--- 1. Loading Dataset ---")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, usecols=[1])
data = df.values.astype('float32')

```


```python
# 2. Visualize dữ liệu gốc
print("\n--- 2. Visualizing Raw Data ---")
plt.figure(figsize=(10, 6))
plt.plot(df, label='Monthly Passengers')
plt.title('International Airline Passengers')
plt.xlabel('Month (Index)')
plt.ylabel('Passengers')
plt.legend()
plt.show()

```


```python
# 3. Tiền xử lý (Scaling & Windowing)
print("\n--- 3. Preprocessing ---")
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data)

train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

```


```python
# 4. Xây dựng mô hình LSTM cải tiến
print("\n--- 4. Building LSTM Model ---")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(1, look_back)),
    LSTM(50),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

```


```python
# 5. Huấn luyện (Training)
print("\n--- 5. Training Model ---")
# Tăng số epoch để mô hình học tốt hơn
history = model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0, validation_split=0.1)

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM Training History')
plt.legend()
plt.show()

```


```python
# 6. Dự báo & Đánh giá (Evaluation)
print("\n--- 6. Predicting & Evaluating ---")
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY_orig = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY_orig = scaler.inverse_transform([testY])

from sklearn.metrics import mean_squared_error
import math
trainScore = math.sqrt(mean_squared_error(trainY_orig[0], trainPredict[:,0]))
print(f'Train Score: {trainScore:.2f} RMSE')
testScore = math.sqrt(mean_squared_error(testY_orig[0], testPredict[:,0]))
print(f'Test Score: {testScore:.2f} RMSE')

plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(dataset), label='Actual Data')
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
plt.plot(trainPredictPlot, label='Train Prediction')
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
plt.plot(testPredictPlot, label='Test Prediction')
plt.title('Airline Passengers Prediction with LSTM')
plt.legend()
plt.show()

```


```python
# 7. Dự báo tương lai (Future Forecasting)
print("\n--- 7. Forecasting next 12 months ---")
last_window = dataset[-look_back:].reshape(1, 1, look_back)
future_predictions = []

current_window = last_window
for i in range(12):
    pred = model.predict(current_window)
    future_predictions.append(pred[0, 0])
    # Update window: shift left and append new prediction
    new_val = pred
    current_window = np.array([np.append(current_window[0, 0, 1:], new_val)]).reshape(1, 1, look_back)

future_predictions = scaler.inverse_transform([future_predictions])
print("Forecasted values for next 12 months:\n", future_predictions[0])

# Vẽ biểu đồ bao gồm cả dự báo tương lai
plt.figure(figsize=(12, 6))
full_data = scaler.inverse_transform(dataset)
plt.plot(range(len(full_data)), full_data, label='Historical Data')
plt.plot(range(len(full_data), len(full_data) + 12), future_predictions[0], 'r--', label='Future Forecast')
plt.title('Airline Passengers - 12 Months Forecast')
plt.legend()
plt.show()
```


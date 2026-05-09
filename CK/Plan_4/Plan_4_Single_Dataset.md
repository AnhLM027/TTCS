# KẾ HOẠCH BÁO CÁO THỰC TẬP CƠ SỞ (KẾ HOẠCH 4 - 1 DATASET XUYÊN SUỐT)

**Tổng quan yêu cầu:**

* **Thời lượng:** 4 Chương, mỗi chương ~12 trang. Tổng ~50 trang.
* **Nội dung:** Lịch sử phát triển, Lý thuyết chuyên sâu, Code & Giải thích từng dòng map với lý thuyết.
* **Mục tiêu Plan 4:** Sử dụng **DUY NHẤT 1 dataset xuyên suốt**, áp dụng nhiều mô hình khác nhau (CNN, RNN, LSTM) để:
  * So sánh hiệu năng
  * Thể hiện rõ bản chất từng kiến trúc
  * Tăng tính logic và liên kết giữa các chương

---

# PHẦN 1: DATASET XUYÊN SUỐT

## 📊 Dataset: IMDB Movie Review Sentiment Analysis (Kaggle)

* **Link:** [https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format)
* **Bài toán:** Phân loại cảm xúc văn bản (Positive / Negative)
* **Quy mô:** ~50,000 mẫu
* **Dạng dữ liệu:** Chuỗi văn bản (text)

---

## 🎯 Lý do chọn dataset này

* ✔ Phù hợp cho **CNN, RNN, LSTM**
* ✔ Dữ liệu chuẩn, phổ biến, dễ giải thích
* ✔ Có thể dùng chung xuyên suốt 4 chương
* ✔ Dễ demo (input text → output sentiment)
* ✔ Rất phù hợp để “giải thích từng dòng code”

---

# PHẦN 2: CẤU TRÚC CHI TIẾT TỪNG CHƯƠNG (~12 Trang/Chương)

---

# CHƯƠNG 1: TRÍ TUỆ NHÂN TẠO & TIỀN XỬ LÝ DỮ LIỆU

## 1.1. Lịch sử phát triển & Tầm quan trọng (2 trang)

* Tổng quan AI, Machine Learning, Deep Learning
* Vai trò của dữ liệu trong AI

---

## 1.2. Lý thuyết tiền xử lý dữ liệu văn bản (3 trang)

* Text Cleaning (lowercase, remove punctuation)
* Tokenization
* Stopwords Removal
* Padding & Sequence
* Vector hóa:
  * Bag of Words
  * TF-IDF
  * Word Embedding

---

## 1.3. Thực nghiệm: Tiền xử lý IMDB Dataset (7 trang)

* Load dataset
* Làm sạch dữ liệu
* Chuyển text → số
* Padding

👉 Giải thích từng dòng code:
* vì sao cần tokenize
* vì sao cần padding
* mapping sang input model

---

# CHƯƠNG 2: MẠNG NƠ-RON TÍCH CHẬP (CNN)

## 2.1. Lịch sử phát triển CNN (2 trang)

* LeNet → AlexNet → VGG
* Ứng dụng trong Computer Vision & NLP

---

## 2.2. Lý thuyết CNN cho dữ liệu văn bản (3 trang)

* Conv1D trên text
* Kernel (filter) = n-gram extractor
* Pooling
* Fully Connected

---

## 2.3. Thực nghiệm: Text Classification bằng CNN (7 trang)

### Model:
```text
Embedding → Conv1D → MaxPooling → Dense
```

👉 Nội dung:
* Xây dựng model CNN cho text
* Train & evaluate

👉 Giải thích code:
* Embedding layer làm gì
* Conv1D trích đặc trưng như thế nào
* MaxPooling giữ thông tin gì

---

# CHƯƠNG 3: MẠNG NƠ-RON HỒI QUY (RNN)

## 3.1. Lịch sử & động lực RNN (2 trang)

* Xử lý dữ liệu chuỗi
* Hạn chế của CNN với chuỗi dài

---

## 3.2. Lý thuyết RNN (3 trang)

* Hidden state
* Time step
* Công thức lan truyền
* Vanishing Gradient

---

## 3.3. Thực nghiệm: Text Classification bằng RNN (7 trang)

### Model:
```text
Embedding → SimpleRNN → Dense
```

👉 Nội dung:
* Huấn luyện RNN trên IMDB

👉 Giải thích:
* cách dữ liệu đi qua từng timestep
* hidden state cập nhật thế nào

---

# CHƯƠNG 4: LONG SHORT-TERM MEMORY (LSTM)

## 4.1. Sự tiến hóa từ RNN → LSTM (2 trang)

* Vấn đề của RNN
* Giải pháp của LSTM

---

## 4.2. Kiến trúc LSTM (3 trang)

* Input gate
* Forget gate
* Output gate
* Memory cell

👉 Giải thích bằng lời + sơ đồ

---

## 4.3. Thực nghiệm: Text Classification bằng LSTM (7 trang)

### Model:
```text
Embedding → LSTM → Dense
```

👉 Nội dung:
* Train LSTM trên cùng dataset

👉 Giải thích:
* LSTM nhớ thông tin dài hạn như thế nào
* so sánh với RNN

---

# 📊 PHẦN BONUS (CỰC KỲ ĂN ĐIỂM)

## So sánh mô hình

| Model | Accuracy | Nhận xét |
| ----- | -------- | -------- |
| CNN   | ...      | nhanh    |
| RNN   | ...      | nhớ ngắn |
| LSTM  | ...      | tốt nhất |

---

# 🎯 KẾ T LUẬN

**Plan 4 giúp bạn:**
* ✔ Không bị rối dataset
* ✔ So sánh được các mô hình
* ✔ Viết logic xuyên suốt 50 trang
* ✔ Dễ giải thích code → đúng yêu cầu đề

---

# 🧠 Câu chốt để ghi vào báo cáo

> Báo cáo sử dụng một tập dữ liệu duy nhất (IMDB) để triển khai và so sánh hiệu quả của các mô hình CNN, RNN và LSTM trong bài toán phân loại cảm xúc văn bản.

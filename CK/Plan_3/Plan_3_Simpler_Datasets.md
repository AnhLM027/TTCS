# KẾ HOẠCH BÁO CÁO THỰC TẬP CƠ SỞ (KẾ HOẠCH 3 - DATASET ĐƠN GIẢN, TRỰC QUAN)

**CẬP NHẬT YÊU CẦU CHÍNH THỨC (Theo Requirements.txt):**
1. **Cấu trúc:** Viết thành đúng 4 Chương: (1) AI-Tiền xử lý dữ liệu, (2) CNN, (3) RNN, (4) LSTM.
2. **Độ dài:** Mỗi chương ~12 trang, tổng cộng ~50 trang, bắt buộc bao gồm **1 trang Tài liệu tham khảo**.
3. **Dữ liệu:** Mỗi chương chọn 1 tập dữ liệu thực tế từ Kaggle hoặc Stanford để thể hiện.
4. **Giải thích Code:** Bắt buộc trình bày kiến thức hiểu biết trong **từng dòng lệnh code** và đối chiếu nó tương ứng với lý thuyết như thế nào.
5. **Lịch sử:** Chú ý trình bày lịch sử phát triển cụ thể (năm ra đời, thuật toán gần đây, và ứng dụng).

*Kế hoạch 3 dưới đây đã được tinh chỉnh 100% để đáp ứng tuyệt đối các tiêu chí chấm điểm này.*

---

## PHẦN 1: ĐỀ XUẤT DATASET CƠ BẢN

### 1. Chương 1 (Tiền xử lý): Titanic - Machine Learning from Disaster (Kaggle)
- **Bài toán:** Dự đoán khả năng sống sót của hành khách.
- **Điểm nhấn:** Bài toán kinh điển nhất trong Machine Learning. Thể hiện các kỹ thuật tiền xử lý cơ bản và dễ hiểu (điền Missing Data cho độ tuổi, mã hóa One-Hot Encoding cho giới tính, Feature Scaling cho giá vé).

### 2. Chương 2 (CNN): Fashion MNIST (hoặc MNIST)
- **Bài toán:** Phân loại hình ảnh quần áo / chữ số viết tay.
- **Điểm nhấn:** Bộ dữ liệu nhẹ, kích thước nhỏ (28x28), không cần cấu hình máy tính mạnh. Rất dễ trực quan hóa hình ảnh gốc và các Feature Map (kết quả sau từng lớp Convolution) để giải thích lý thuyết.

### 3. Chương 3 (RNN): IMDB Movie Reviews Sentiment Analysis
- **Bài toán:** Phân loại cảm xúc (Tích cực/Tiêu cực) của các bài đánh giá phim.
- **Điểm nhấn:** Bài toán NLP cơ bản. Xử lý chuỗi văn bản đơn giản (Tokenization, Padding, Embedding), dễ dàng giải thích luồng hoạt động của mạng RNN với chuỗi văn bản.

### 4. Chương 4 (LSTM): Dự báo số lượng hành khách (Airline Passengers Prediction)
- **Bài toán:** Dự đoán số lượng hành khách đi máy bay trong các tháng tiếp theo dựa trên dữ liệu lịch sử (Dataset kinh điển: International Airline Passengers).
- **Điểm nhấn:** Dữ liệu chỉ có 1 cột số duy nhất theo thời gian, có tính chu kỳ (seasonality) và xu hướng (trend) cực kỳ rõ ràng. Rất dễ huấn luyện (chỉ mất vài giây) và biểu đồ kết quả rất đẹp, trơn tru, dễ giải thích hơn nhiều so với dự báo giá cổ phiếu (thường bị nhiễu loạn).

---

## PHẦN 2: CẤU TRÚC CHI TIẾT TỪNG CHƯƠNG (~12 Trang/Chương)

### CHƯƠNG 1: TRÍ TUỆ NHÂN TẠO & TIỀN XỬ LÝ DỮ LIỆU
**1.1. Lịch sử phát triển & Tầm quan trọng (2 trang)**
**1.2. Lý thuyết các kỹ thuật Tiền xử lý (3 trang)**
- Missing Values, Encoding (Label/One-hot), Feature Scaling cơ bản.
**1.3. Thực nghiệm: Titanic Survival Prediction (7 trang)**
- Code đơn giản hóa, sử dụng biểu đồ EDA trực quan (VD: biểu đồ sinh tồn theo giới tính). Giải thích chi tiết luồng xử lý trước khi đưa vào mô hình.

### CHƯƠNG 2: MẠNG NƠ-RON TÍCH CHẬP (CNN)
**2.1. Lịch sử phát triển CNN (2 trang)** (Tập trung vào LeNet, AlexNet).
**2.2. Lý thuyết cấu trúc CNN (3 trang)** (Convolution, Pooling, Flatten, Fully Connected).
**2.3. Thực nghiệm: Phân loại Fashion MNIST (7 trang)**
- Tự xây dựng mô hình CNN từ đầu (Build from scratch) thay vì dùng Transfer Learning phức tạp. Chèn các hình ảnh minh họa bộ lọc (filters) nhận diện góc cạnh như thế nào.

### CHƯƠNG 3: MẠNG NƠ-RON HỒI QUY (RNN)
**3.1. Lịch sử và Động lực ra đời của RNN (2 trang)**
**3.2. Cấu trúc và Toán học của RNN (3 trang)** (Cấu trúc chuỗi vòng, điểm yếu Vanishing Gradient).
**3.3. Thực nghiệm: Phân tích cảm xúc IMDB (7 trang)**
- Code tiền xử lý text thành vector. Giải thích quá trình RNN nhận từng từ vào hidden state theo từng mốc thời gian (time-steps).

### CHƯƠNG 4: LONG SHORT-TERM MEMORY (LSTM)
**4.1. Sự tiến hóa từ RNN lên LSTM (2 trang)**
**4.2. Kiến trúc cốt lõi của LSTM (3 trang)** (Input gate, Output gate, Forget gate - tập trung giải thích bằng lời văn dễ hiểu thay vì thuần công thức).
**4.3. Thực nghiệm: Dự báo lượng hành khách hàng không (7 trang)**
- Dùng Pandas load data đơn giản (chỉ 1 cột). Trực quan hóa đường số lượng thực tế và dự đoán trên biểu đồ. Nhấn mạnh việc LSTM học được tính chu kỳ lặp lại theo mùa và xu hướng tăng dần theo năm.

---

## PHẦN 3: CHIẾN LƯỢC TRIỂN KHAI VÀ "ĂN ĐIỂM CAO"

Plan bạn chọn là **rất chuẩn để ăn điểm cao** rồi 👍 — giờ mình giúp bạn “nâng cấp” từng bài toán thành **có chiều sâu + có cái để viết 12 trang/chương**.

Mình sẽ không viết lại plan, mà đi thẳng vào **ý tưởng triển khai + điểm ăn điểm + cái để bạn phân tích**.

---

### 🔥 CHƯƠNG 1 – Titanic (Tiền xử lý)

#### 🎯 Mục tiêu chính
👉 Không phải model → mà là **preprocessing + EDA**

#### 💡 Ý tưởng triển khai
**1. EDA (cực kỳ quan trọng)**
* Tỷ lệ sống sót theo:
  * Giới tính
  * Hạng vé (Pclass)
  * Độ tuổi
* Biểu đồ:
  * Countplot
  * Barplot
  * Heatmap correlation

👉 Câu ăn điểm:
> "Female có tỷ lệ sống cao hơn → phản ánh chính sách ưu tiên"

**2. Xử lý dữ liệu (viết rất nhiều được)**
* Missing:
  * Age → median theo Pclass
  * Embarked → mode
* Encoding:
  * Sex → Label
  * Embarked → One-hot
* Feature engineering:
  * FamilySize = SibSp + Parch
  * IsAlone

👉 Đây là phần giúp bạn đủ 12 trang dễ nhất

**3. Model đơn giản**
* Logistic Regression / Decision Tree

👉 Không cần deep learning ở đây

#### 🧠 Điểm ăn điểm
* Giải thích **tại sao phải chuẩn hóa dữ liệu**
* Giải thích **ảnh hưởng feature đến kết quả**

---

### 🔥 CHƯƠNG 2 – CNN (Fashion MNIST)

#### 🎯 Mục tiêu
👉 Hiểu **Convolution hoạt động thế nào**

#### 💡 Ý tưởng triển khai
**1. Visualize dữ liệu**
* Show 10 ảnh mẫu
* Label distribution

**2. Xây CNN từ đầu**
```python
Conv2D → ReLU → MaxPooling → Flatten → Dense
```

**3. Visualize cực ăn điểm**
* **Feature map**
  * Lấy output của Conv layer
  * Hiển thị filter học được

👉 Câu ăn điểm:
> "CNN học được edge, texture trước khi nhận diện object"

**4. So sánh model**
* CNN vs MLP (Fully Connected)

👉 Cho thấy CNN tốt hơn

#### 🧠 Điểm ăn điểm
* Giải thích:
  * Kernel là gì
  * Stride / Padding
* Visual filter → giảng viên rất thích

---

### 🔥 CHƯƠNG 3 – RNN (IMDB)

#### 🎯 Mục tiêu
👉 Hiểu **xử lý chuỗi**

#### 💡 Ý tưởng triển khai
**1. Tiền xử lý text**
* Cleaning
* Tokenization
* Padding

👉 Map với lý thuyết sequence

**2. Model RNN**
```python
Embedding → SimpleRNN → Dense
```

**3. Visualize (rất quan trọng)**
* WordCloud
* Label distribution
* Learning curve

**4. Phân tích lỗi**
* Ví dụ:
  * câu dài → RNN fail
  * sarcasm → sai

👉 Câu ăn điểm:
> "RNN gặp vấn đề vanishing gradient nên khó nhớ dài hạn"

#### 🧠 Điểm ăn điểm
* Giải thích:
  * hidden state
  * time-step
* Minh họa dòng chảy dữ liệu

---

### 🔥 CHƯƠNG 4 – LSTM (Time Series)

#### 🎯 Mục tiêu
👉 Hiểu **memory dài hạn**

#### 💡 Ý tưởng triển khai
**1. Visualize dữ liệu**
* Line chart theo thời gian
* Nhận xét:
  * Trend
  * Seasonality

**2. Chuẩn bị dữ liệu**
* Sliding window:
```text
[1,2,3] → 4
[2,3,4] → 5
```

**3. Model LSTM**
```python
LSTM → Dense
```

**4. Visualize kết quả**
* Actual vs Predicted

👉 Câu ăn điểm:
> "LSTM học được chu kỳ mùa vụ"

**5. So sánh**
* RNN vs LSTM

👉 LSTM tốt hơn rõ ràng

#### 🧠 Điểm ăn điểm
* Giải thích:
  * Forget gate
  * Input gate
  * Output gate

---

### 🔥 BONUS – KẾT LUẬN CUỐI BÀI (CỰC QUAN TRỌNG)

Bạn nên thêm 1 bảng so sánh:

| Model | Ưu điểm       | Nhược điểm       | Ứng dụng    |
| ----- | ------------- | ---------------- | ----------- |
| CNN   | Xử lý ảnh tốt | Không hiểu chuỗi | CV          |
| RNN   | Hiểu chuỗi    | Quên nhanh       | NLP         |
| LSTM  | Nhớ dài       | Chậm             | Time series |

---

### 🎯 CHIẾN LƯỢC ĂN ĐIỂM CAO

👉 Đừng cố làm khó, hãy:
* Dataset đơn giản
* Nhưng:
  * Visual nhiều
  * Giải thích sâu
  * So sánh rõ

---

### 🔥 KẾT LUẬN THẲNG

Plan này:
> ✅ Rất phù hợp người mới
> ✅ Dễ code
> ✅ Dễ viết 50 trang
> ✅ Dễ ăn điểm cao

# KẾ HOẠCH BÁO CÁO THỰC TẬP CƠ SỞ (KẾ HOẠCH 1 - 4 DATASET ĐỘC LẬP)

**Tổng quan yêu cầu:**
- **Thời lượng:** 4 Chương, mỗi chương ~12 trang. Tổng ~50 trang.
- **Nội dung:** Lịch sử phát triển, Lý thuyết chuyên sâu, Code & Giải thích từng dòng map với lý thuyết.
- **Yêu cầu đặc biệt:** Sử dụng dataset từ Kaggle/Stanford.

---

## PHẦN 1: ĐỀ XUẤT DATASET ĐỘC ĐÁO & CHẤT LƯỢNG CAO

### 1. Chương 1 (Tiền xử lý): Kaggle - Home Credit Default Risk
- **Bài toán:** Dự đoán khả năng vỡ nợ của khách hàng (Credit Scoring).
- **Điểm nhấn:** Thể hiện được kỹ năng Feature Engineering, kỹ thuật xử lý Missing Data nâng cao (KNN Imputer), và xử lý Imbalanced data (SMOTE/ADASYN).

### 2. Chương 2 (CNN): Kaggle - Cassava Leaf Disease Classification
- **Bài toán:** Phân loại bệnh trên lá cây sắn (Nông nghiệp thông minh).
- **Điểm nhấn:** Thể hiện được Data Augmentation phức tạp, Transfer Learning (ResNet, EfficientNet) thay vì chỉ tự build CNN cơ bản.

### 3. Chương 3 (RNN): Kaggle - ECG Heartbeat Categorization (MIT-BIH Arrhythmia)
- **Bài toán:** Phân tích tín hiệu điện tâm đồ (ECG) để phân loại rối loạn nhịp tim.
- **Điểm nhấn:** Dùng RNN cho Tín hiệu chuỗi thời gian 1D (1D Time Series Signal) trong y tế tạo ra sự mới mẻ.

### 4. Chương 4 (LSTM): NASA Turbofan Engine Degradation Simulation
- **Bài toán:** Predictive Maintenance - Dự đoán thời gian sống còn lại (RUL) của động cơ.
- **Điểm nhấn:** Bài toán cốt lõi trong Công nghiệp 4.0. Giải quyết vấn đề Vanishing Gradient của RNN với chuỗi cảm biến dài.

---

## PHẦN 2: CẤU TRÚC CHI TIẾT TỪNG CHƯƠNG (~12 Trang/Chương)

### CHƯƠNG 1: TRÍ TUỆ NHÂN TẠO & TIỀN XỬ LÝ DỮ LIỆU
**1.1. Lịch sử phát triển & Tầm quan trọng (2 trang)**
**1.2. Lý thuyết các kỹ thuật Tiền xử lý (3 trang)**
- Missing Values, Encoding, Outlier Detection, Feature Scaling.
**1.3. Ứng dụng thực nghiệm: Home Credit Default Risk (7 trang)**
- Code xử lý missing data và imbalanced data. Giải thích map với lý thuyết phân phối.

### CHƯƠNG 2: MẠNG NƠ-RON TÍCH CHẬP (CNN)
**2.1. Lịch sử phát triển CNN (2 trang)** (Neocognitron -> LeNet-5 -> AlexNet -> ResNet).
**2.2. Lý thuyết cấu trúc CNN (3 trang)** (Convolutional, Pooling, Fully Connected).
**2.3. Thực nghiệm: Phân loại bệnh lá cây sắn (7 trang)**
- Code định nghĩa model, Data Augmentation. Giải thích ý nghĩa của Filter/Kernel trong việc trích xuất đặc trưng biên (edge).

### CHƯƠNG 3: MẠNG NƠ-RON HỒI QUY (RNN)
**3.1. Lịch sử và Động lực ra đời của RNN (2 trang)**
**3.2. Cấu trúc và Toán học của RNN (3 trang)** (Unrolled architecture, Hidden State, Vanishing Gradient).
**3.3. Thực nghiệm: Phân loại nhịp tim ECG (7 trang)**
- Code SimpleRNN layer. Giải thích sự truyền đi của Hidden state qua các time steps.

### CHƯƠNG 4: LONG SHORT-TERM MEMORY (LSTM)
**4.1. Sự tiến hóa từ RNN lên LSTM (2 trang)**
**4.2. Kiến trúc cốt lõi của LSTM (3 trang)** (Forget Gate, Input Gate, Output Gate).
**4.3. Thực nghiệm: Predictive Maintenance - NASA Turbofan (7 trang)**
- Code LSTM layer. Phân tích kết quả dự đoán thời gian sống còn lại của động cơ (RUL).

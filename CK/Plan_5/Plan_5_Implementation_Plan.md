# KẾ HOẠCH TRIỂN KHAI PLAN_5: HỆ THỐNG NGHIÊN CỨU AI CHUYÊN SÂU
*(Data Processing - CNN - RNN - LSTM)*

## I. TỔNG QUAN CHIẾN LƯỢC
- **Mục tiêu:** Xây dựng 4 chương nghiên cứu độc lập nhưng có tính kế thừa, mỗi chương là một dự án nghiên cứu nhỏ (Mini Research Project).
- **Cấu trúc thư mục:**
    - `Plan_5/code/ch1_data_processing/`: Modules, Notebook, Results.
    - `Plan_5/code/ch2_cnn/`: Models, Visualization, Notebook.
    - `Plan_5/code/ch3_rnn/`: Sequence Analysis, Notebook.
    - `Plan_5/code/ch4_lstm/`: Attention Mechanism, Forecasting, Notebook.
- **Phong cách trình bày:** Học thuật + Thực chiến (Đầy đủ bảng so sánh, biểu đồ phân tích sâu).

---

## II. NỘI DUNG CHI TIẾT TỪNG CHƯƠNG

### CHƯƠNG 1: XỬ LÝ DỮ LIỆU (THE FOUNDATION OF INTELLIGENCE)
*Mục tiêu: Chứng minh ảnh hưởng của tiền xử lý đến hiệu suất mô hình.*
- **Datasets (2 bộ đơn giản nhất):** 
    1. **Titanic (Tabular):** Missing values & Encoding.
    2. **Breast Cancer (Numerical):** Outliers & Scaling.
- **Thử nghiệm:** So sánh Performance (Accuracy/F1) của 1 mô hình trên dữ liệu Gốc vs dữ liệu đã qua Tiền xử lý.
- **Kỹ thuật:** SMOTE, Robust Scaler, Target Encoding, PCA, t-SNE visualization.
- **Visualization:** Missing data matrix, Outlier boxplots, PCA 2D/3D cluster, Feature distribution.

### CHƯƠNG 2: MẠNG NƠ-RON TÍCH CHẬP (DEEP VISION & FEATURES)
*Mục tiêu: Hiểu cách CNN trích xuất đặc trưng và sức mạnh của Transfer Learning.*
- **Datasets (2 bộ đơn giản nhất):** 1. MNIST, 2. Fashion-MNIST.
- **Mô hình (4 kiến trúc):** 
    1. **Simple CNN** (Dưới 3 lớp).
    2. **LeNet-5** (Kiến trúc kinh điển).
    3. **ResNet-18** (Transfer Learning).
    4. **VGG-Small** (Deep architecture).
- **Phân tích sâu:** 
    - **Feature Map Visualization:** Xem các filter học được những gì (cạnh, khối, texture).
    - **Grad-CAM:** Giải thích vùng nhìn của AI trên ảnh y tế.
- **Visualization:** Training/Val loss, Accuracy curve, Filter visualization, Grad-CAM Heatmap.

### CHƯƠNG 3: MẠNG NƠ-RON HỒI QUY (TEMPORAL DEPENDENCIES)
*Mục tiêu: Phân tích khả năng ghi nhớ chuỗi và vấn đề Vanishing Gradient.*
- **Datasets (2 bộ đơn giản nhất):** 1. IMDB Sentiment, 2. Sine Wave (Simple Seq).
- **Mô hình (4 kiến trúc):** 
    1. **Vanilla RNN.**
    2. **Bidirectional RNN.**
    3. **GRU** (Gated Recurrent Unit).
    4. **Deep RNN.**
- **Phân tích sâu:** 
    - Vẽ đồ thị Gradient để thấy hiện tượng vanishing gradient.
    - Tokenization & Embedding visualization (t-SNE).
- **Visualization:** Hidden state heatmap, Prediction vs Actual (Sequence), WordCloud.

### CHƯƠNG 4: LSTM & ATTENTION MECHANISM (LONG-TERM MEMORY)
*Mục tiêu: Tối ưu hóa dự báo chuỗi thời gian và cơ chế tập trung.*
- **Datasets (2 bộ đơn giản nhất):** 1. Google Stock, 2. Weather Forecasting.
- **Mô hình (4 kiến trúc):** 
    1. **Standard LSTM.**
    2. **Stacked LSTM.**
    3. **Bi-LSTM.**
    4. **LSTM + Attention.**
- **Phân tích sâu:** 
    - **Attention Weight Visualization:** Xem model tập trung vào mốc thời gian nào trong quá khứ.
    - Multi-step prediction (Dự báo nhiều bước trong tương lai).
- **Visualization:** Attention Heatmap, Sliding window analysis, Error analysis chart, Multi-step forecast plot.

---

## III. QUY TRÌNH THỰC THI (PIPELINE)
Mỗi chương sẽ tuân thủ quy trình:
1. **Load Data:** Tải từ nguồn chính thống (Sklearn/Kaggle).
2. **Exploratory Data Analysis (EDA):** Phân tích kỹ phân phối, tương quan.
3. **Preprocessing:** Làm sạch, mã hóa, chuẩn hóa.
4. **Modeling:** Xây dựng và huấn luyện 3-4 model.
5. **Evaluation:** Dùng Confusion Matrix, ROC-AUC, PR-Curve.
6. **Comparison:** Lập bảng so sánh Metrics (Time, Params, Acc).
7. **In-depth Analysis:** Giải thích code, giải thích biểu đồ và rút ra kết luận nghiên cứu.

---

## IV. CÁC PHẦN MỞ RỘNG (BONUS)
- **Hyperparameter Tuning:** Sử dụng Optuna.
- **Explainable AI:** SHAP cho Tabular, Grad-CAM cho Image.
- **Deployment Demo:** File code `app.py` để demo sản phẩm thực tế cho mỗi chương.

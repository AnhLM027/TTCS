# KẾ HOẠCH TRIỂN KHAI PLAN_6: HỆ THỐNG NGHIÊN CỨU & BÁO CÁO THỰC TẬP AI CHUYÊN SÂU
*(Data Processing - CNN - RNN - LSTM & Attention)*

## I. TỔNG QUAN CHIẾN LƯỢC
- **Mục tiêu:** Xây dựng 4 chương nghiên cứu thực nghiệm độc lập nhưng có tính kế thừa. Mỗi chương đáp ứng đồng thời 2 nhiệm vụ:
    1. **Nghiên cứu chuyên sâu:** Đối đầu với thách thức thực tế qua 4 Dataset và 4 Model mỗi chương (Tổng cộng 16 Dataset và 16 Model).
    2. **Báo cáo học thuật:** Trình bày đầy đủ Lý thuyết, Lịch sử và Phân tích code chi tiết (đáp ứng chuẩn báo cáo thực tập).
- **Quy mô báo cáo:** Dự kiến **100 - 150+ trang** (đảm bảo độ phủ kiến thức và thực nghiệm cực lớn).
- **Cấu trúc mỗi chương:** Lý thuyết/Lịch sử -> Thực nghiệm chuyên sâu 4-4-4 -> Phân tích Code & Benchmarking.
- **Phong cách trình bày:** Học thuật + Thực chiến (Đầy đủ bảng so sánh, biểu đồ phân tích sâu, giải thích từng dòng lệnh).
- **Cấu trúc thư mục:** Chia theo chương (`Plan_6/Chương_n/`) gồm `code/`, `data/`, `docs/`.
- **Nguồn tài liệu:** Nội dung báo cáo (`docs/`) được kế thừa và chỉnh sửa từ `Plan_3/final`.

---

## II. NỘI DUNG CHI TIẾT TỪNG CHƯƠNG

### CHƯƠNG 1: AI VÀ TIỀN XỬ LÝ DỮ LIỆU (ADVANCED PREPROCESSING)
- **Lý thuyết & Lịch sử:**
    - Trình bày tổng quan về Trí tuệ nhân tạo (AI), các lĩnh vực ứng dụng.
    - Vai trò của tiền xử lý dữ liệu trong Machine Learning / Deep Learning.
    - Các kỹ thuật tiền xử lý phổ biến: làm sạch dữ liệu, chuẩn hóa, mã hóa nhãn, chia tập train/test,...
- **Thực nghiệm chuyên sâu (4 Datasets):** 
    1. **Used Car Price (Numerical):** Xử lý Outliers & Feature Derivation.
    2. **Telco Churn (Tabular):** Class Imbalance (SMOTE) & Encoding.
    3. **Credit Card Fraud (Anomaly):** Precision-Recall focus trên dữ liệu cực nhiễu.
    4. **Adult Census (Categorical):** Xử lý biến hạng mục phức tạp (Target Encoding).
- **Kỹ thuật trọng tâm:** RobustScaler, IQR Outlier detection, Feature Engineering (Car Age, Mileage per Year), SMOTE.
- **Phân tích Code:** Phân tích chi tiết **từng dòng lệnh trong code**, giải thích đoạn code đó liên hệ với lý thuyết như thế nào.
- **Visualization:** Missing data matrix, Correlation Heatmap, Boxplots (trước/sau xử lý), PCA 2D/3D cluster.

### CHƯƠNG 2: MẠNG NƠ-RON TÍCH CHẬP (CNN - DEEP VISION)
- **Lý thuyết & Lịch sử:**
    - Trình bày khái niệm, cấu trúc và nguyên lý hoạt động của CNN.
    - Lịch sử phát triển: Ra đời năm nào, các mô hình nổi bật (**LeNet, AlexNet, VGG, ResNet**...).
    - Các ứng dụng thực tế của CNN trong nhận dạng ảnh, y tế, xe tự lái,...
- **Thực nghiệm chuyên sâu (4 Models):** Custom CNN, MobileNetV2, ResNet-18, EfficientNet-B0.
- **Datasets (4 bộ):** Intel Image, Flowers-102, Chest X-Ray, Natural Images.
- **Kỹ thuật trọng tâm:** Transfer Learning, Fine-tuning, Data Augmentation (Flip, Rotation, GridMask), Batch Normalization.
- **Phân tích Code:** Phân tích chi tiết code xây dựng mô hình, giải thích các lớp (Convolution, Pooling, FC) liên hệ với lý thuyết.
- **Visualization:** Grad-CAM (Giải thích vùng nhìn của AI), Feature Maps, Confusion Matrix, ROC-AUC.

### CHƯƠNG 3: MẠNG NƠ-RON HỒI TIẾP (RNN - TEMPORAL CONTEXT)
- **Lý thuyết & Lịch sử:**
    - Trình bày khái niệm, cơ chế hoạt động của RNN.
    - Ưu điểm, hạn chế (**Vanishing Gradient**, short-term memory,...).
    - Ứng dụng trong xử lý ngôn ngữ tự nhiên, chuỗi thời gian,...
- **Thực nghiệm chuyên sâu (4 Models):** Vanilla RNN, Bidirectional GRU, Stacked LSTM, Conv1D + RNN.
- **Datasets (4 bộ):** Stock Sentiment, IMDB Reviews, Reuters News, Amazon Reviews.
- **Kỹ thuật trọng tâm:** Word Embeddings (Word2Vec, GloVe), Padding/Truncating, Gradient Clipping, Dropout.
- **Phân tích Code:** Phân tích chi tiết dòng lệnh, giải thích cơ chế Hidden State và xử lý chuỗi trong code.
- **Visualization:** WordCloud theo nhãn cảm xúc, Token embedding t-SNE, Gradient Flow plot.

### CHƯƠNG 4: MẠNG LSTM & CƠ CHẾ ATTENTION (LONG-TERM MEMORY)
- **Lý thuyết & Lịch sử:**
    - Trình bày khái niệm LSTM, cấu trúc các cổng (**Input, Forget, Output Gate**).
    - Lý do LSTM ra đời để cải tiến RNN (giải quyết Long-term dependencies).
    - Các ứng dụng: Dự báo giá cổ phiếu, dịch máy, chatbot,...
- **Thực nghiệm chuyên sâu (4 Models):** Standard LSTM, Bi-LSTM, LSTM + Attention, Transformer (Encoder).
- **Datasets (4 bộ):** Beijing Air Quality, Google Stock Price, Energy Consumption, HAR (Sensor data).
- **Kỹ thuật trọng tâm:** Multi-step Forecasting, Sliding Window, Attention Score calculation, Teacher Forcing.
- **Phân tích Code:** Phân tích chi tiết cách các cổng hoạt động trong code Python, trực quan hóa cơ chế Attention.
- **Visualization:** Attention Weight Heatmap, Forecast vs Actual (Multi-step), Error distribution.

---

## III. QUY TRÌNH THỰC THI & TRÌNH BÀY (DUNG LƯỢNG 100 - 150 TRANG)
1.  **Chi tiết hóa lý thuyết:** Không chỉ nêu khái niệm mà phải phân tích lịch sử và ưu nhược điểm sâu sắc ở đầu mỗi chương.
2.  **Phân tích Code (Line-by-Line):** Phân tích chi tiết **từng dòng lệnh**, giải thích đoạn code đó liên hệ với lý thuyết như thế nào.
3.  **Benchmarking đa chiều:** Lập bảng so sánh 4 mô hình về Accuracy, Time, Params, Memory Usage cho từng chương.
4.  **Minh họa tối đa:** Đầy đủ kết quả thực nghiệm, biểu đồ, hình ảnh trực quan hóa sâu cho từng kỹ thuật được sử dụng.
5.  **Tài liệu tham khảo:** 01 trang danh mục chuẩn (IEEE/APA) ở cuối báo cáo.

---

## IV. CÁC THÀNH PHẦN MỞ RỘNG (RESEARCH BONUS)
- **Optuna:** Tự động tối ưu hóa siêu tham số để tìm model tốt nhất cho báo cáo.
- **Explainable AI:** Sử dụng SHAP và Grad-CAM để tăng tính thuyết phục cho phần phân tích.
- **Deployment:** Demo Web App bằng Streamlit tích hợp toàn bộ kết quả nghiên cứu.

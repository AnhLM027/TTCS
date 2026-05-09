# BÁO CÁO TỔNG KẾT DỰ ÁN NGHIÊN CỨU AI CHUYÊN SÂU (PLAN_5)

## 1. TỔNG QUAN DỰ ÁN
Dự án này là một chuỗi các nghiên cứu thực nghiệm nhằm khám phá bản chất của các kiến trúc học máy và học sâu hiện đại. Báo cáo tập trung vào việc đối chiếu lý thuyết với thực nghiệm thông qua 4 chương trọng tâm.

---

## CHƯƠNG 1: XỬ LÝ DỮ LIỆU - NỀN TẢNG CỦA TRÍ TUỆ NHÂN TẠO
*Tập trung: Phân tích ảnh hưởng của Preprocessing đến hiệu năng mô hình.*

### 1.1. Mục tiêu nghiên cứu
Chứng minh rằng việc tiền xử lý dữ liệu đúng cách có thể cải thiện đáng kể hiệu suất mô hình, thậm chí quan trọng hơn việc lựa chọn mô hình phức tạp.

### 1.2. Danh sách Dataset thực nghiệm
- **Titanic:** Xử lý dữ liệu khuyết thiếu và mã hóa phân loại.
- **Breast Cancer:** Xử lý Outlier và chuẩn hóa đặc trưng.
- **SMS Spam:** Tiền xử lý văn bản (Cleaning, Tokenization).
- **Dry Bean:** Giảm chiều dữ liệu và trực quan hóa cụm.

### 1.3. Phương pháp nghiên cứu
- Thực hiện Pipeline: Clean -> Scale -> Encode -> Select Features.
- So sánh kết quả mô hình Baseline trước và sau khi xử lý.
- Phân tích trực quan bằng PCA và t-SNE.

---

## CHƯƠNG 2: MẠNG NƠ-RON TÍCH CHẬP (CNN) - TRÍ TUỆ THỊ GIÁC
*Tập trung: Cơ chế trích xuất đặc trưng và Giải thích vùng nhìn.*

### 2.1. Kiến trúc thử nghiệm
- So sánh các mạng từ nông đến sâu: LeNet, ResNet, VGG.
- Phân tích sự khác biệt giữa Global Average Pooling và Flatten.

### 2.2. Kỹ thuật giải thích (Interpretable CV)
- Sử dụng **Grad-CAM** để trực quan hóa những gì CNN thực sự "thấy".

---

## CHƯƠNG 3: MẠNG NƠ-RON HỒI QUY (RNN) - MÔ HÌNH HÓA CHUỖI
*Tập trung: Phụ thuộc thời gian và Vấn đề Vanishing Gradient.*

### 3.1. Bài toán thực nghiệm
- Phân tích cảm xúc (Sentiment Analysis) và Dự báo chuỗi đơn giản.

### 3.2. Phân tích bộ nhớ
- Đối chiếu khả năng ghi nhớ của Vanilla RNN so với GRU/LSTM.

---

## CHƯƠNG 4: LSTM & ATTENTION - BỘ NHỚ DÀI HẠN & SỰ TẬP TRUNG
*Tập trung: Dự báo phức tạp và Cơ chế Attention.*

### 4.1. Dự báo chuỗi thời gian đa biến
- Áp dụng trên dữ liệu Chứng khoán và Thời tiết.

### 4.2. Cơ chế Attention
- Trực quan hóa bản đồ chú ý để hiểu cách mô hình chọn lọc thông tin quan trọng.

---
*(Báo cáo này sẽ được cập nhật số liệu thực nghiệm sau khi chạy code)*

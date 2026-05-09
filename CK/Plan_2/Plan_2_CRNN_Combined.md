# KẾ HOẠCH BÁO CÁO THỰC TẬP CƠ SỞ (KẾ HOẠCH 2 - BÀI TOÁN KẾT HỢP CRNN XUYÊN SUỐT)

**Tổng quan định hướng:**
Sử dụng **một bài toán duy nhất** xuyên suốt cả Báo cáo để tạo sự liên kết chặt chẽ. Bài toán này sẽ yêu cầu kiến trúc kết hợp cả mạng tích chập (CNN) và mạng hồi quy (RNN/LSTM) hay còn gọi là kiến trúc **Encoder-Decoder (CRNN)**. 
Bài toán được chọn: **Image Captioning (Sinh văn bản mô tả hình ảnh)** bằng dataset `Flickr8k` hoặc `MS COCO`.

---

## PHẦN 1: TẠI SAO LẠI CHỌN IMAGE CAPTIONING?
- **Khẳng định tầm vóc:** Kết hợp 2 lĩnh vực khó nhất là Computer Vision (CNN) và Natural Language Processing (NLP/LSTM).
- **Tính logic cho cấu trúc 4 chương:**
  - Chương 1: Tiền xử lý Ảnh (cho CNN) và Tiền xử lý Văn bản (cho LSTM).
  - Chương 2: Xây dựng Encoder (CNN) để hiểu bức ảnh.
  - Chương 3 & 4: Xây dựng Decoder (RNN/LSTM) để học chuỗi văn bản và ráp nối lại sinh ra câu mô tả.
- **Vượt trội so với mặt bằng chung:** Khác hẳn với các bài tutorial rời rạc, cách này mô phỏng quy trình xây dựng AI đa phương thức (Multimodal AI) như cách ChatGPT hay Midjourney hoạt động nội bộ.

---

## PHẦN 2: CẤU TRÚC CHI TIẾT TỪNG CHƯƠNG (~12 Trang/Chương)

### CHƯƠNG 1: TRÍ TUỆ NHÂN TẠO & TIỀN XỬ LÝ ĐA PHƯƠNG THỨC (MULTIMODAL)
**1.1. Lịch sử phát triển & Tầm quan trọng (2 trang)**
- Từ AI đơn lẻ đến xu hướng AI Đa phương thức (Multimodal).
**1.2. Lý thuyết Tiền xử lý Ảnh & Văn bản (3 trang)**
- Ảnh: Resize, Normalization, Data Augmentation.
- Văn bản: Tokenization, Padding, Word Embeddings (Chuyển chữ thành số).
**1.3. Ứng dụng: Tiền xử lý cho Flickr8k (7 trang)**
- **Mã nguồn & Giải thích:**
  - Code tạo từ điển (Vocabulary), gán nhãn `<start>`, `<end>`.
  - Code Tokenizer của Keras/PyTorch. Giải thích: *Máy tính không hiểu chữ "Dog", dòng code này chuyển "Dog" thành vector số để mạng nơ-ron có thể tính toán toán học.*

### CHƯƠNG 2: MẠNG NƠ-RON TÍCH CHẬP (CNN) - TRÍCH XUẤT ẢNH (ENCODER)
**2.1. Lịch sử phát triển CNN (2 trang)**
- Các mô hình nổi tiếng như VGG, Inception, ResNet. Khái niệm Transfer Learning.
**2.2. Lý thuyết cấu trúc CNN (3 trang)**
- Cách CNN nhìn nhận ảnh 2D qua Convolution và Pooling.
**2.3. Thực nghiệm: Xây dựng Image Encoder (7 trang)**
- Dùng ResNet50 (hoặc VGG16) loại bỏ lớp cuối (Classification Layer) để lấy Vector Đặc trưng (Feature Vector).
- **Mã nguồn & Giải thích:**
  - Code load Transfer Learning Model. Giải thích: *Thay vì train lại từ đầu (mất nhiều ngày), dòng lệnh `include_top=False` giúp ta lấy lại bộ não đã được train trên hàng triệu ảnh của Google/Microsoft, biến bức ảnh 2D thành 1 ma trận số 1D đại diện cho toàn bộ vật thể trong ảnh.*

### CHƯƠNG 3: MẠNG NƠ-RON HỒI QUY (RNN) - MÔ HÌNH NGÔN NGỮ
**3.1. Lịch sử và Động lực ra đời của RNN (2 trang)**
- Tại sao CNN không xử lý được chuỗi (sequence)? Sự ra đời của kiến trúc mạng có "trí nhớ".
**3.2. Cấu trúc và Toán học của RNN (3 trang)**
- Backpropagation through time (BPTT). Vấn đề Vanishing Gradient khi câu văn quá dài.
**3.3. Thực nghiệm: Tạo bộ khung sinh từ (Word Generator) (7 trang)**
- Cách RNN học sự phân bố xác suất của từ tiếp theo dựa vào các từ trước đó.
- Code `Embedding Layer` và `SimpleRNN`. Nhấn mạnh vào lý do SimpleRNN thất bại với câu mô tả ảnh quá dài, tạo tiền đề dẫn sang LSTM.

### CHƯƠNG 4: LSTM & RÁP NỐI MÔ HÌNH HOÀN CHỈNH (IMAGE CAPTIONING)
**4.1. Sự tiến hóa từ RNN lên LSTM (2 trang)**
- Cơ chế Memory Cell và 3 Gates (Forget, Input, Output).
**4.2. Kiến trúc Encoder-Decoder (3 trang)**
- Cách ghép nối Vector ảnh (từ Chương 2) vào làm State khởi tạo cho LSTM (từ Chương 4).
**4.3. Thực nghiệm: Huấn luyện & Sinh câu mô tả (7 trang)**
- Ráp model CNN và LSTM lại với nhau.
- **Mã nguồn & Giải thích:**
  - Code hàm Inference sinh từ (Greedy Search hoặc Beam Search).
  - Đưa 1 bức ảnh lạ vào và show kết quả model tự động in ra câu mô tả bằng tiếng Anh. Giải thích: *Tại mỗi step, LSTM mở cổng Input Gate nhận từ cũ và đặc trưng ảnh, mở Output Gate nhả ra từ mới tiếp theo cho đến khi gặp thẻ `<end>`.*

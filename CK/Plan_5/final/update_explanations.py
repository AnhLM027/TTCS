import os
import re

def add_detailed_explanations(file_path):
    """Add detailed explanations to all figures in a LaTeX file"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to find figures with captions
    figure_pattern = r'(\\begin\{figure\}\[H\]\s*\\centering\s*\\includegraphics\[width=[\d\.\w]+\]\{[^}]+\}\s*\\caption\{[^}]+\}\s*\\end\{figure\})'

    def replace_figure(match):
        figure_text = match.group(1)

        # Extract caption to determine explanation
        caption_match = re.search(r'\\caption\{([^}]+)\}', figure_text)
        if not caption_match:
            return figure_text

        caption = caption_match.group(1)

        # Generate detailed explanation based on caption keywords
        explanation = ""

        if 'thiết lập môi trường' in caption.lower() or 'khởi tạo các thư viện' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Đoạn code này thiết lập nền tảng cho toàn bộ quá trình phân tích dữ liệu. Các thư viện được import theo chuẩn PEP 8, đảm bảo code dễ đọc và bảo trì. Việc cấu hình môi trường đúng cách giúp tránh conflicts giữa các phiên bản thư viện và đảm bảo reproducibility của experiments."

        elif 'nạp dữ liệu' in caption.lower() and 'titanic' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Phương thức df.info() cung cấp cái nhìn tổng quan về cấu trúc dữ liệu: số lượng dòng, cột, kiểu dữ liệu và missing values. df.head() hiển thị sample data để verify format. Đây là bước EDA cơ bản, giúp phát hiện sớm các vấn đề về data quality trước khi preprocessing."

        elif 'trực quan hóa dữ liệu thiếu' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Sử dụng thư viện Missingno để tạo heatmap trực quan hóa pattern missing values. Seaborn correlation heatmap đo lường mối quan hệ tuyến tính giữa variables. Việc này giúp phân loại missing data (MCAR/MAR/MNAR) và lựa chọn imputation strategy phù hợp."

        elif 'bản đồ nhiệt dữ liệu thiếu' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Heatmap sử dụng missingno library trực quan hóa missing patterns. Mỗi ô trắng là giá trị có sẵn, đen là missing. Việc sắp xếp theo thứ tự cho thấy pattern có cấu trúc, giúp classify missing data types và design appropriate handling strategies."

        elif 'tỷ lệ sống sót theo hạng ghế' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Biểu đồ cột cho thấy survival rate giảm từ hạng 1 (63\\%) xuống hạng 3 (24\\%). Phản ánh cấu trúc xã hội Titanic nơi upper class được ưu tiên evacuation. Đây là ví dụ về non-linear relationship trong data, đòi hỏi model nắm bắt pattern này."

        elif 'phân phối độ tuổi' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Histogram + KDE cho thấy phân phối tuổi right-skewed, đỉnh tại 20-30 tuổi. Việc dùng median (28) thay mean để impute phù hợp vì median robust với outliers. Áp dụng thực tế của statistical theory trong missing value treatment."

        elif 'ma trận tương quan' in caption.lower() and 'titanic' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Correlation matrix với Pearson coefficient từ -1 đến 1. Pclass có correlation âm mạnh với Survived (-0.34), Fare và Pclass âm (-0.55). Giúp identify multicollinearity và support feature selection decisions."

        elif 'triển khai quy trình tiền xử lý' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Pipeline preprocessing hoàn chỉnh với SimpleImputer (median), OneHotEncoder, StandardScaler. Pipeline của sklearn đảm bảo consistency giữa train/test sets, tránh data leakage. Đây là best practice trong machine learning engineering."

        elif 'so sánh hiệu suất' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Biểu đồ so sánh accuracy giữa raw data (70\\%) và processed data (82\\%). Chứng minh giá trị của preprocessing: transform 'dirty' data thành format phù hợp cho algorithms. Tuy nhiên, preprocessing không phải magic - chỉ optimize data cho models."

        elif 'nạp bộ dữ liệu breast cancer' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Wisconsin Breast Cancer dataset từ sklearn với 569 samples, 30 numeric features mô tả tumor attributes. Chuyển đổi sang DataFrame tận dụng pandas methods. Target binary encoded (0:malignant, 1:benign), phù hợp cho binary classification tasks."

        elif 'biểu đồ boxplot' in caption.lower() and 'outliers' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Boxplot visualize scale differences giữa features. 'mean area' range 143-2501, 'mean smoothness' 0.05-0.16. Outliers theo IQR rule. Chứng minh necessity của scaling: nếu không normalize, 'mean area' sẽ dominate Euclidean distance calculations."

        elif 'ma trận tương quan breast cancer' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Heatmap reveal feature clusters với high correlation: radius/perimeter/area (r=0.99). Multicollinearity issue cho Linear models. Feature selection hoặc PCA cần thiết để reduce dimensionality và improve model stability."

        elif 'phân phối các đặc trưng' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} KDE plots chồng lên nhau show distribution differences giữa malignant/benign. Malignant có higher mean radius. Chứng minh discriminative power của features, support classification model design. Class-conditional distribution analysis giúp understand data patterns."

        elif 'mối quan hệ giữa radius và nhãn mục tiêu' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Scatter plot với color coding theo class show mean radius là strong discriminator: malignant (orange) có higher radius than benign (blue). Though overlap exists, clear separation visible. Explain why radius commonly used in tumor diagnosis - malignant cells typically larger."

        elif 'khởi tạo các thư viện deep learning' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Setup technology stack cho deep learning. TensorFlow 2.x + Keras provide high-level neural network building interface. Numpy handle tensor operations, matplotlib/seaborn create visualizations. 'Agg' backend avoid GUI conflicts on headless servers."

        elif 'nạp dữ liệu mnist' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} Keras MNIST loader auto-download và cache dataset. Reshape (60000, 28, 28, 1) prepare cho CNN input. Normalization x/255.0 scale pixels từ [0,255] về [0,1], help gradient descent converge faster vì weights init thường quanh 0."

        elif 'trực quan hóa dữ liệu mnist' in caption.lower():
            explanation = "\\textbf{Giải thích chi tiết:} EDA code sử dụng matplotlib plot sample images, class distribution histogram, pixel intensity distribution. Class balance checking quan trọng vì imbalance bias model. Pixel distribution show most pixels are 0 (background), help understand data sparsity."

        # Add explanation after figure if not already present
        if not re.search(r'\\textbf\{Giải thích chi tiết:\}', figure_text):
            return figure_text + "\n\n" + explanation
        else:
            return figure_text

    # Apply replacements
    new_content = re.sub(figure_pattern, replace_figure, content)

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Updated {file_path}")

# Process all chapter files
chapters = ['ch1_preprocessing', 'ch2_cnn', 'ch3_rnn', 'ch4_lstm']
base_path = '/home/llm/AnhLM/TTCS/CK/Plan_5/final/chapters/'

for chapter in chapters:
    file_path = f"{base_path}{chapter}.tex"
    if os.path.exists(file_path):
        add_detailed_explanations(file_path)
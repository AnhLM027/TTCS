import kagglehub
import shutil
import os

# Định nghĩa các dataset cho từng chương
datasets = {
    "Chương_1": [
        ("austinreese/craigslist-car-trucks-data", "used_car_price"),
        ("blastchar/telco-customer-churn", "telco_churn"),
        ("mlg-ulb/creditcardfraud", "credit_card_fraud"),
        ("uciml/adult-census-income", "adult_census")
    ],
    "Chương_2": [
        ("puneet6060/intel-image-classification", "intel_images"),
        ("alxmamaev/flowers-recognition", "flowers"),
        ("paultimothymooney/chest-xray-pneumonia", "chest_xray"),
        ("prasunroy/natural-images", "natural_images")
    ],
    "Chương_3": [
        ("yash612/stockmarket-sentiment-dataset", "stock_sentiment"),
        ("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews", "imdb_reviews"),
        ("bittu/amazon-reviews-sentiment-analysis", "amazon_reviews")
    ],
    "Chương_4": [
        ("uciml/beijing-air-quality-dataset", "beijing_air_quality"),
        ("rahulsah06/google-stock-price-prediction", "google_stock"),
        ("robikscube/hourly-energy-consumption", "energy_consumption"),
        ("uciml/human-activity-recognition-with-smartphones", "har_sensor")
    ]
}

def download_and_move():
    base_dir = "Plan_6"
    for chapter, items in datasets.items():
        chapter_data_dir = os.path.join(base_dir, chapter, "data")
        os.makedirs(chapter_data_dir, exist_ok=True)
        
        for repo, folder_name in items:
            print(f"--- Downloading {repo} to {chapter}/{folder_name} ---")
            try:
                # Tải dataset
                path = kagglehub.dataset_download(repo)
                dest_path = os.path.join(chapter_data_dir, folder_name)
                
                # Di chuyển dữ liệu vào thư mục mong muốn
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                
                # Nếu là file đơn lẻ, tạo thư mục và copy vào
                if os.path.isfile(path):
                    os.makedirs(dest_path, exist_ok=True)
                    shutil.copy(path, dest_path)
                else:
                    shutil.copytree(path, dest_path)
                
                print(f"Successfully moved to {dest_path}")
            except Exception as e:
                print(f"Error downloading {repo}: {e}")

if __name__ == "__main__":
    download_and_move()

import kagglehub
import os
import shutil

def download_and_move(handle, target_dir):
    print(f"Downloading {handle}...")
    try:
        path = kagglehub.dataset_download(handle)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        # Move all files from downloaded path to target_dir
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(target_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print(f"Moved {handle} to {target_dir}")
    except Exception as e:
        print(f"Error downloading {handle}: {e}")

# Missing datasets fix
datasets_to_fix = [
    ("bittu/amazon-reviews-sentiment-analysis", "Plan_6/Chương_3/data/amazon_reviews"),
    ("uciml/beijing-air-quality-dataset", "Plan_6/Chương_4/data/beijing_air_quality"),
    ("rahulsatyam/google-stock-price-dataset", "Plan_6/Chương_4/data/google_stock_price")
]

for handle, target in datasets_to_fix:
    download_and_move(handle, target)

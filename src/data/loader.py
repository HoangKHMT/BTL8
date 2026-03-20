import pandas as pd
import os

def load_raw_data():
    # Tự động tìm đường dẫn file csv từ vị trí file loader.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "../../data/raw/creditcard.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"✅ Đã nạp dữ liệu: {df.shape}")
        return df
    else:
        print(f"❌ Không tìm thấy file tại: {file_path}")
        return None
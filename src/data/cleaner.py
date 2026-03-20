import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_and_preprocess(df):
    """Làm sạch và chuẩn hóa dữ liệu chuyên sâu"""
    df = df.drop_duplicates()
    # Chỉ fillna cho các cột số
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    scaler = StandardScaler()
    # Kiểm tra cột tồn tại trước khi scale
    if 'Amount' in df.columns:
        df['std_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    if 'Time' in df.columns:
        df['std_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    return df.drop(columns=[c for c in ['Time', 'Amount'] if c in df.columns])

def preprocess_for_mining(df):
    """Tiền xử lý riêng cho khai phá luật kết hợp"""
    df_mining = df.copy()
    
    # 1. Xác định cột tiền để chia nhóm (Binning)
    potential_cols = ['std_amount', 'normAmount', 'Amount']
    target_col = next((c for c in potential_cols if c in df_mining.columns), None)
    
    if target_col:
        # Chia mức chi tiêu thành 3 nhóm nhãn chữ
        df_mining['Amount_Bin'] = pd.qcut(df_mining[target_col], q=3, labels=['Low', 'Medium', 'High'])
    
    # 2. Chuyển Class sang nhãn chữ Status
    if 'Class' in df_mining.columns:
        df_mining['Status'] = df_mining['Class'].map({0: 'Normal', 1: 'Status_FRAUD'})
    
    # 3. CHỈ GIỮ LẠI các cột phân loại (nhãn chữ) để chạy Apriori
    # Loại bỏ hoàn toàn V1-V28 và các cột số thực để tránh lỗi ValueError
    keep_features = ['Amount_Bin', 'Status']
    df_final = df_mining[keep_features]
        
    return df_final
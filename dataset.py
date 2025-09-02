import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

class AirQualityDataset(Dataset):
    def __init__(self, csv_file, seq_length=24):
        self.data = pd.read_csv(csv_file)
        self.seq_length = seq_length
        self.scaler = StandardScaler()
        
        # 篩選古亭站數據
        self.data = self.data[self.data['site_name'] == '古亭'].reset_index(drop=True)
        
        # 特徵選擇與預處理
        features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'day_of_week', 'is_holiday', 'so2', 'no2', 'co', 'o3']
        # 風向分解為 sin 和 cos
        self.data['wind_dir_sin'] = np.sin(np.radians(self.data['wind_direction']))
        self.data['wind_dir_cos'] = np.cos(np.radians(self.data['wind_direction']))
        features += ['wind_dir_sin', 'wind_dir_cos']
        
        # 處理缺失值（簡單方法：用平均值填補）
        self.data[features] = self.data[features].fillna(self.data[features].mean())
        self.data['pm25'] = self.data['pm25'].fillna(self.data['pm25'].mean())
        
        self.X = self.data[features].values
        self.y = self.data['pm25'].values
        self.X = self.scaler.fit_transform(self.X)
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.X[idx:idx+self.seq_length]  # [seq_length, num_features]
        y = self.y[idx+self.seq_length]      # 單一值（下一小時 PM2.5）
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_scaler(csv_file):
    data = pd.read_csv(csv_file)
    data = data[data['site_name'] == '古亭'].reset_index(drop=True)
    scaler = StandardScaler()
    features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'day_of_week', 'is_holiday', 'so2', 'no2', 'co', 'o3']
    data['wind_dir_sin'] = np.sin(np.radians(data['wind_direction']))
    data['wind_dir_cos'] = np.cos(np.radians(data['wind_direction']))
    features += ['wind_dir_sin', 'wind_dir_cos']
    data[features] = data[features].fillna(data[features].mean())
    scaler.fit(data[features].values)
    return scaler
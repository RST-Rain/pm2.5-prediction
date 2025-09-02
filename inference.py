import torch
import pandas as pd
import numpy as np
from model import AirQualityTransformer
from dataset import get_scaler

def inference(csv_file, start_idx, seq_length=24):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AirQualityTransformer(input_dim=12).to(device)
    model.load_state_dict(torch.load('runs/best_model.pth'))
    model.eval()

    # 讀取數據並標準化
    data = pd.read_csv(csv_file)
    data = data[data['site_name'] == '古亭'].reset_index(drop=True)
    scaler = get_scaler(csv_file)
    features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'day_of_week', 'is_holiday', 'so2', 'no2', 'co', 'o3']
    data['wind_dir_sin'] = np.sin(np.radians(data['wind_direction']))
    data['wind_dir_cos'] = np.cos(np.radians(data['wind_direction']))
    features += ['wind_dir_sin', 'wind_dir_cos']
    x = data[features].values[start_idx:start_idx+seq_length]
    x = scaler.transform(x)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    # 推論
    with torch.no_grad():
        output = model(x)
        print(f'Predicted PM2.5 concentration: {output.item():.2f} μg/m³')

if __name__ == '__main__':
    csv_file = 'data/val.csv'
    start_idx = 0
    inference(csv_file, start_idx)
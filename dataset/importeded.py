import pandas as pd
import numpy as np

# 讀取 CSV 檔案
df = pd.read_csv('C:/Users/user/Desktop/小專題/pm2.5-prediction/dataset/air_quality_guting_combined.csv')

# 檢查初始數據是否有缺失值
print("初始數據缺失值檢查：")
print(df.isnull().sum())

# 將 timestamp 轉換為 datetime 格式，並處理可能的錯誤
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M', errors='coerce')

# 提取日期部分作為新欄位
df['date'] = df['timestamp'].dt.date

# 檢查數值欄位的數據類型並轉換為數值型，忽略錯誤
numeric_columns = ['pm2.5', 'so2', 'no2', 'co', 'o3', 'pm10']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 根據日期分組，計算每個數值欄位的平均值
daily_avg = df.groupby('date').agg({
    'pm2.5': 'mean',
    'so2': 'mean',
    'no2': 'mean',
    'co': 'mean',
    'o3': 'mean',
    'pm10': 'mean'
}).reset_index()

# 無條件捨去到小數點後一位
for col in numeric_columns:
    daily_avg[col] = np.floor(daily_avg[col] * 10) / 10

# 填補可能的 NaN 值（例如用前一個有效值填補或設為 0，根據需求選擇）
daily_avg = daily_avg.fillna(0)

# 將日期轉回字符串格式
daily_avg['date'] = daily_avg['date'].astype(str)

# 輸出結果到控制台
print("處理後的每日平均值：")
print(daily_avg)

# 保存為新的 CSV 檔案
daily_avg.to_csv('C:/Users/user/Desktop/小專題/pm2.5-prediction/dataset/air_quality_guting_combined_daily_average.csv', index=False)
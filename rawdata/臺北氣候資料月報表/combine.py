import pandas as pd
import os
import glob
import re

pd.set_option('display.max_rows', None)

def combine_weather_data(input_dir, output_file):
    # 讀取所有 CSV 檔案 (466920-YYYY-MM.csv)
    files = glob.glob(os.path.join(input_dir, "466920-*.csv"))
    print("找到的檔案：", files)
    dfs = []
    
    for file in files:
        # 讀取 CSV，假設第一行是中文標題，第二行是英文標題
        df = pd.read_csv(file, skiprows=1)  # 跳過第一行中文標題
        
        # 從檔名擷取年份與月份
        match = re.search(r'(\d{4})-(\d{2})\.csv$', file)
        if match:
            year, month = match.groups()
            df["year"] = year
            df["month"] = month
        
        dfs.append(df)
    
    # 合併所有檔案
    data = pd.concat(dfs, ignore_index=True)
    
    # 整理日期欄位
    data['month'] = data['month'].astype(str).str.zfill(2)
    data['ObsTime'] = data['ObsTime'].astype(str).str.zfill(2)

    # 生成 timestamp
    data['timestamp'] = pd.to_datetime(data['year'] + '-' + data['month'] + '-' + data['ObsTime'], errors="coerce")

    # 選擇需要的欄位並重新命名
    output_cols = [
        'timestamp', 
        'Temperature',  # 氣溫(℃)
        'T Max',       # 最高氣溫(℃)
        'T Min',       # 最低氣溫(℃)
        'RH',          # 相對溼度(%)
        'WS',          # 風速(m/s)
        'WD',          # 風向(360degree)
        'Precp',       # 降水量(mm)
        'SunShine',    # 日照時數(hour)
        'GloblRad',    # 全天空日射量(MJ/㎡)
        'UVI Max'      # 日最高紫外線指數
    ]
    
    # 確保所有需要的欄位存在
    data = data[output_cols]
    
    # 將數值欄位轉換為數值型，處理可能的非數值（如 'T'）
    numeric_cols = ['Temperature', 'T Max', 'T Min', 'RH', 'WS', 'WD', 'Precp', 'SunShine', 'GloblRad', 'UVI Max']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # 按時間排序
    data = data.sort_values('timestamp')
    
    # 保存到 CSV
    data.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Generated {output_file} with {len(data)} rows")

if __name__ == '__main__':
    input_dir = r"./"
    output_file = r"../../dataset/taipei_weather_combined.csv"
    
    combine_weather_data(input_dir, output_file)

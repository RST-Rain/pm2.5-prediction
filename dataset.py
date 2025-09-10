import pandas as pd
import numpy as np
import os

# -----------------------------
# 1️⃣ 定義 CSV 檔案路徑
file_name = r'./data/air_quality_guting_combined.csv'

# -----------------------------
# 2️⃣ 讀取 CSV
df = pd.read_csv(file_name)

# -----------------------------
# 3️⃣ 自動找日期欄位
date_column = None
for col in df.columns:
    try:
        tmp = pd.to_datetime(df[col], errors='coerce')
        if tmp.notnull().sum() / len(tmp) > 0.5:
            date_column = col
            break
    except:
        continue

if date_column is None:
    raise ValueError("⚠️ 找不到日期欄位，請確認 CSV 裡有日期欄！")

print(f"✅ 自動偵測到日期欄位：'{date_column}'")

# -----------------------------
# 4️⃣ 將日期欄位轉成 datetime
df['timestamp'] = pd.to_datetime(df[date_column], errors='coerce')

# -----------------------------
# 5️⃣ 檢查轉換失敗的日期
if df['timestamp'].isnull().any():
    print("⚠️ 有些日期轉換失敗，原始值如下：")
    print(df[df['timestamp'].isnull()][date_column])

# -----------------------------
# 6️⃣ 提取日期
df['date'] = df['timestamp'].dt.date

# -----------------------------
# 7️⃣ 數值欄位轉為數值型
numeric_columns = ['pm2.5', 'so2', 'no2', 'co', 'o3', 'pm10']
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        df[col] = 0  # 欄位不存在就填0

# -----------------------------
# 8️⃣ 每日平均值
daily_avg = df.groupby('date')[numeric_columns].mean().reset_index()

# 9️⃣ 無條件捨去到小數點後一位
for col in numeric_columns:
    daily_avg[col] = np.floor(daily_avg[col] * 10) / 10

# 10️⃣ 填補 NaN
daily_avg = daily_avg.fillna(0)

# 11️⃣ 日期轉回字串
daily_avg['date'] = daily_avg['date'].astype(str)

# -----------------------------
# 12️⃣ 輸出結果（確保資料夾存在）
output_dir = './dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, 'air_quality_guting_combined_daily_average.csv')
daily_avg.to_csv(output_file, index=False)

print(f"✅ 已處理完成，保存到 {output_file}")

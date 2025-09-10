import json
import pandas as pd
import csv

# 假設檔案名為 data.json
with open("地面測站每日雨量資料-每日雨量-2024.json", "r", encoding="utf-8") as f:
    data = json.load(f)

records = []

# 進入 JSON 結構
locations = data["cwaopendata"]["resources"]["resource"]["data"]["surfaceObs"]["location"]

for loc in locations:
    # 只取 StationName == "臺北"
    if loc["station"]["StationName"] == "臺北":
        for obs in loc["stationObsTimes"]["stationObsTime"]:
            date = obs["Date"]
            precp = obs["weatherElements"]["Precipitation"]
            # 嘗試轉成浮點數 (處理 "T"、"X" 這類無效值)
            try:
                precp = float(precp)
            except:
                precp = 0
            records.append({"Date": date, "Precipitation": precp})

# 存成 DataFrame
df = pd.DataFrame(records)

print(df)

with open('../../dataset/taipei_rainfall_combined.csv', 'w', newline='', encoding='utf-8') as csvfile:
    df.to_csv(csvfile, index=False)

from combine_2025 import combine_rainfall_data
combine_rainfall_data('地面測站每日雨量資料-每日雨量-2025.json')
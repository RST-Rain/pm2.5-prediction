import json
import pandas as pd
from datetime import datetime
import os
import requests

def fetch_weather_data(url):
    params = {
        "Authorization": "CWA-47E9E15F-F236-47CC-A7D9-0F49DE9968EF"
    }
    headers = {
        "accept": "application/json"
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print("抓取失敗，錯誤代碼:", response.status_code)
        return None

def get_value(row, key, alt_values, avg):
    #預報值
    if key in row and pd.notna(row[key]):
        print(f"{key}: Using forecast value {row[key]}")
        return row[key]

    #即時值
    if key in alt_values and pd.notna(alt_values.get(key)):
        print(f"{key}: Using real-time value {alt_values[key]}")
        return alt_values[key]

    #歷史資料 (比對同月同日)
    if key in avg.columns:
        try:
            target_date = pd.to_datetime(row["date"])   # 預測日期

            same_day = avg[pd.to_datetime(avg["timestamp"]).dt.strftime("%m-%d") == target_date.strftime("%m-%d")]
            if not same_day.empty:
                value = same_day[key].iloc[-1]  # 拿最近一年的同月日
                print(f"{key}: Using historical same-day value {value}")
                return value
        except Exception as e:
            print(f"{key}: historical lookup failed, {e}")

    #預設值
    print(f"{key}: Using default 0.0")
    return 0.0

def safe_float(val, default=0.0):
    try:
        val = float(val)
        if val == -990:   # 缺測值轉成 0
            return default
        return val
    except:
        return default


def predict_future():
    # 抓預報
    url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-D0047-061"
    today_str = datetime.today().strftime("%Y-%m-%d")
    predict_dir = "./weather_prediction/"
    os.makedirs(predict_dir, exist_ok=True)

    filename = f"{today_str}_未來三日預報.json"
    filepath = os.path.join(predict_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(fetch_weather_data(url), f, ensure_ascii=False, indent=2)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 只取出大安區
    locations = data["records"]["Locations"]
    daan_data = None
    for loc in locations:
        for area in loc["Location"]:
            if area["LocationName"] == "大安區":
                daan_data = area
                break

    if not daan_data:
        raise ValueError("找不到大安區的資料")

    target_map = {
        "溫度": "Temperature",
        "降雨": "Precipitation",
        "相對濕度": "RH",
        "風速": "WS",
        "風向": "WD",
        "日照時數": "SunShine",
        "全天空日射量": "GloblRad"
    }

    records = []
    for elem in daan_data["WeatherElement"]:
        if elem["ElementName"] in target_map:
            indicator = target_map[elem["ElementName"]]
            for t in elem["Time"]:
                try:
                    date = t["DataTime"][:10]
                except:
                    date = t["StartTime"][:10]
                value_dict = t["ElementValue"][0]
                if "WeatherDescription" in value_dict:
                    continue
                value = list(value_dict.values())[0]
                try:
                    value = float(value)
                except:
                    wind_map = {
                        "偏北風": 0.0, "東北風": 45.0, "偏東風": 90.0,
                        "東南風": 135.0, "偏南風": 180.0, "西南風": 225.0,
                        "偏西風": 270.0, "西北風": 315.0,
                    }
                    value = wind_map.get(value, 0.0)
                records.append({"date": date, "indicator": indicator, "value": value})

    df = pd.DataFrame(records)
    daily_avg = df.groupby(["date", "indicator"])["value"].mean().reset_index()
    result = daily_avg.pivot(index="date", columns="indicator", values="value").reset_index()
    print("Forecast data columns:", result.columns.tolist())

    # 即時缺失資料抓取
    url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0003-001"
    new_data = fetch_weather_data(url)
    if not new_data:
        raise ValueError("無法抓取即時資料")


    latest_data = None
    for loc in new_data["records"]["Station"]:
        if loc["StationId"] == "466920":
            latest_data = loc
            break

    if not latest_data:
        raise ValueError("找不到 Taipei 站 (466920) 的資料")

    we = latest_data["WeatherElement"]
    alt_values = {
        "Temperature": safe_float(we.get("AirTemperature", 0)),
        "Precipitation": safe_float(we.get("Now", {}).get("Precipitation", 0)),
        "RH": safe_float(we.get("RelativeHumidity", 0)),
        "WS": safe_float(we.get("WindSpeed", 0)),
        "WD": safe_float(we.get("WindDirection", 0)),
        "SunShine": safe_float(we.get("SunshineDuration", 0)),
        "GloblRad": safe_float(we.get("GloblRad", 0)) if "GloblRad" in we else None,
    }

    with open('./data/taipei_weather_combined.csv', 'r', encoding='utf-8') as f:
        avg = pd.read_csv(f)

    # 預測
    from inference import predict_air_pollution
    model_path = './model.pth'
    scaler_path = './scaler.pkl'

    for _, row in result.iterrows():
        input_data = pd.DataFrame({
            'Temperature': [get_value(row, 'Temperature', alt_values, avg)],
            'Precipitation': [get_value(row, 'Precipitation', alt_values, avg)],
            'RH': [get_value(row, 'RH', alt_values, avg)],
            'WS': [get_value(row, 'WS', alt_values, avg)],
            'WD': [get_value(row, 'WD', alt_values, avg)],
            'SunShine': [get_value(row, 'SunShine', alt_values, avg)],
            'GloblRad': [get_value(row, 'GloblRad', alt_values, avg)],
        })
        print("Input data:", input_data)
        pred = predict_air_pollution(model_path, scaler_path, input_data)
        print(f"{row['date']} → Predicted PM2.5: {pred:.4f}")

        # === 儲存結果到 CSV ===
        output_file = "./weather_prediction/predicted_pm25.csv"

        save_row = input_data.copy()
        save_row["date"] = row["date"]
        save_row["Predicted_PM25"] = round(pred, 4)

        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            existing = existing[existing["date"] != row["date"]]  # 覆蓋同日期
            updated = pd.concat([existing, save_row], ignore_index=True)
        else:
            updated = save_row
        cols = ['date', 'Temperature', 'Precipitation', 'RH', 'WS', 'WD', 'SunShine', 'GloblRad', 'Predicted_PM25']
        updated = updated[cols]
        updated.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"已更新輸出檔案：{output_file}")

if __name__ == "__main__":
    predict_future()
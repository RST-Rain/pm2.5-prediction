import pandas as pd
import os
import glob

def combine_air_quality_data(input_dir, output_file):
    # 讀取所有 CSV 檔案
    files = glob.glob(os.path.join(input_dir, "空氣品質小時值_臺北市_古亭站*.csv"))
    dfs = []
    
    for file in files:
        df = pd.read_csv(file)
        # 標準化 sitename 和 county
        df['sitename'] = df['sitename'].replace('Guting', '古亭')
        df['county'] = df['county'].replace('Taipei City', '臺北市')
        dfs.append(df)
    
    # 合併所有檔案
    data = pd.concat(dfs, ignore_index=True)
    
    # 篩選古亭站數據
    data = data[data['sitename'] == '古亭'].copy()
    
    # 將污染物數據 pivot 為單行
    pivot_data = data.pivot_table(
        index=['monitordate', 'sitename'],
        columns='itemengname',
        values='concentration',
        aggfunc='first'
    ).reset_index()
    
    # 重新命名欄位
    pivot_data.columns = ['timestamp', 'site_name', 'CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2']
    
    # 按時間排序
    pivot_data['timestamp'] = pd.to_datetime(pivot_data['timestamp'])
    pivot_data = pivot_data.sort_values('timestamp')
    
    # 選擇最終欄位
    output_cols = ['timestamp', 'site_name', 'pm25', 'so2', 'no2', 'co', 'o3', 'pm10']
    pivot_data[output_cols] = pivot_data[['timestamp', 'site_name', 'PM2.5', 'SO2', 'NO2', 'CO', 'O3', 'PM10']]
    
    # 保存到 CSV
    pivot_data[output_cols].to_csv(output_file, index=False)
    print(f"Generated {output_file} with {len(pivot_data)} rows")

if __name__ == '__main__':
    input_dir = r"./"
    output_file = r"../../dataset/air_quality_guting_combined.csv"

    
    combine_air_quality_data(input_dir, output_file)
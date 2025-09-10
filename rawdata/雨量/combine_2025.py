import json
import csv
from datetime import datetime

def combine_rainfall_data(file_name):
    # Read JSON from file
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Find the location with StationName '臺北'
    locations = data['records']['location']
    taipei_data = None
    for loc in locations:
        if loc['station']['StationName'] == '臺北':
            taipei_data = loc
            break

    if taipei_data:
        # Extract daily data up to 2025-08-31
        obs_times = taipei_data['stationObsTimes']['stationObsTime']
        cutoff_date = datetime.strptime('2025-08-31', '%Y-%m-%d')
        
        # Write to CSV
        with open('../../dataset/taipei_rainfall_combined.csv', 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for obs in obs_times:
                obs_date = datetime.strptime(obs['Date'], '%Y-%m-%d')
                if obs_date <= cutoff_date:
                    date = obs['Date']
                    precip = obs['weatherElements'].get('Precipitation', '')
                    if precip == "T" or precip == "X" or precip == "":
                        precip = 0.0
                    writer.writerow([date, precip])
        
        print("CSV file 'taipei_rainfall_combined.csv' created successfully.")
    else:
        print("No data found for StationName '臺北'. Note: In the provided JSON, no station named '臺北' is present; it has '新北warden北', '淡水', '屏東'. You may need to adjust the name if it's a typo.")

if __name__ == "__main__":
    combine_rainfall_data('地面測站每日雨量資料-每日雨量-2025.json')

import json
import csv
from datetime import datetime

# Read JSON from file
with open('地面測站每日雨量資料-每日雨量.json', 'r', encoding='utf-8') as file:
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
    with open('../../dataset/taipei_rainfall_combined.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['StationID', 'StationName', 'Date', 'Precipitation'])
        station_id = taipei_data['station']['StationID']
        station_name = taipei_data['station']['StationName']
        for obs in obs_times:
            obs_date = datetime.strptime(obs['Date'], '%Y-%m-%d')
            if obs_date <= cutoff_date:
                date = obs['Date']
                precip = obs['weatherElements'].get('Precipitation', '')
                writer.writerow([station_id, station_name, date, precip])
    
    print("CSV file 'taipei_rainfall_combined.csv' created successfully.")
else:
    print("No data found for StationName '臺北'. Note: In the provided JSON, no station named '臺北' is present; it has '新北warden北', '淡水', '屏東'. You may need to adjust the name if it's a typo.")
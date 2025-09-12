# 比較不重要的資料
.venv/
__pycache__/
.gitignore
.python-version
pyproject.toml
README.md
uv.lock
model.pth
scaler.pkl


# 重要的資料
runs/     # 訓練過程輸出的紀錄
data/  # 整理過的數據
weather_prediction/    # 未來三天的空氣品質預報

rawdata/  # 原始數據
- 雨量/
- 空氣品質/
- 臺北氣候資料月報表/

inference.py # 單張資料推論，展示成果
main.py    # 主程式，負責呼叫 dataset 與 model，執行訓練與驗證
model.py    # 自訂模型 class，定義 forward 流程
predict_inference.py    #使用最新資料預測
資料來源
古亭站空氣品質: https://data.moenv.gov.tw/dataset/detail/AQX_P_202
地面測站每日雨量資料-每日雨量: https://opendata.cwa.gov.tw/dist/opendata-swagger.html?urls.primaryName=openAPI#/氣候/get_v1_rest_datastore_C_B0025_001
臺北氣候資料月報表:https://codis.cwa.gov.tw/StationData
github連結
https://github.com/RST-Rain/pm2.5-prediction

使用 uv 管理 Python 環境
https://dev.to/codemee/shi-yong-uv-guan-li-python-huan-jing-53hg

Grok頂級輔助過程
https://grok.com/s/bGVnYWN5_4e561423-2376-43ab-8ef7-88cde67fbff0

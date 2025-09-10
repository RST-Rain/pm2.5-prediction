import torch
import pandas as pd
import joblib
from model import AirPollutionModel
from peft import LoraConfig, get_peft_model

def predict_air_pollution(model_path, scaler_path, input_data):
    # -----------------------------
    # 設定裝置
    # -----------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    # -----------------------------
    # 載入 scaler
    # -----------------------------
    scaler = joblib.load(scaler_path)
    input_data_array = input_data.to_numpy()
    input_data_scaled = scaler.transform(input_data_array)

    # -----------------------------
    # 建立原始模型結構
    # -----------------------------
    input_dim = input_data.shape[1]
    base_model = AirPollutionModel(input_dim)

    # -----------------------------
    # 設定 LoRA 配置
    # -----------------------------
    lora_config = LoraConfig(
        r=4,
        lora_alpha=15,
        target_modules=["fc1", "fc2"],
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)

    # -----------------------------
    # 載入 LoRA 權重
    # -----------------------------
    # weights_only=True 避免 pickle 安全問題
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    # -----------------------------
    # 推論
    # -----------------------------
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(input_data_scaled, dtype=torch.float32).to(device)
        outputs = model(inputs)
        predictions = outputs.cpu().squeeze().item()

    return predictions

# -----------------------------
# 範例使用
# -----------------------------
if __name__ == "__main__":
    model_path = "model.pth"
    scaler_path = "scaler.pkl"

    # 假設 new_data 是 DataFrame
    new_data = pd.DataFrame({
        "Temperature": [29.875],      # 氣溫 22°C
        "Precipitation": [8.75],     # 微量降雨 0.2 mm
        "RH": [100],               # 相對濕度 55%
        "WS": [1.25],                # 風速 4.5 m/s
        "WD": [118.125],                 # 東風
        "SunShine": [0],          # 日照 6 小時
        "GloblRad": [14.41]         # 全球輻射 250 W/m²
    })

    pred = predict_air_pollution(model_path, scaler_path, new_data)
    print("預測 PM2.5:", pred)

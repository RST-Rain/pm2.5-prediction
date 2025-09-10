import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import joblib
from model import AirQualityDataset, AirPollutionModel

# LoRA 套件
from peft import LoraConfig, get_peft_model

# -----------------------------
# 自動建立資料夾
# -----------------------------
os.makedirs('runs', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

# -----------------------------
# 檢查檔案是否存在
# -----------------------------
def check_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"⚠️ 檔案不存在：{file_path}")

# -----------------------------
# 載入並預處理資料
# -----------------------------
def load_and_preprocess_data(air_quality_path, rainfall_path, weather_path):
    check_file(air_quality_path)
    check_file(rainfall_path)
    check_file(weather_path)

    air_df = pd.read_csv(air_quality_path)
    rain_df = pd.read_csv(rainfall_path)
    weather_df = pd.read_csv(weather_path)
    
    # 統一日期欄位
    air_df.rename(columns={'date': 'Date'}, inplace=True)
    weather_df.rename(columns={'timestamp': 'Date'}, inplace=True)
    rain_df.rename(columns={'date': 'Date'}, inplace=True)

    # 轉 datetime
    air_df['Date'] = pd.to_datetime(air_df['Date'], errors="coerce")
    rain_df['Date'] = pd.to_datetime(rain_df['Date'], errors="coerce")
    weather_df['Date'] = pd.to_datetime(weather_df['Date'], errors="coerce")

    # 移除無效日期
    air_df.dropna(subset=['Date'], inplace=True)
    rain_df.dropna(subset=['Date'], inplace=True)
    weather_df.dropna(subset=['Date'], inplace=True)

    # 合併資料
    merged_df = pd.merge(air_df, rain_df, on='Date', how='inner')
    merged_df = pd.merge(merged_df, weather_df, on='Date', how='inner')

    # 特徵與目標
    features = ['Temperature', 'Precipitation', 'RH', 'WS', 'WD', 'SunShine', 'GloblRad']
    target = 'pm2.5'

    # 填補缺失值
    merged_df.replace('', np.nan, inplace=True)
    merged_df.fillna(0, inplace=True)

    # 取出特徵與目標
    X = merged_df[features].values
    y = merged_df[target].values

    # 標準化特徵
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler

# -----------------------------
# 計算 Accuracy
# -----------------------------
def calculate_accuracy(predictions, targets, threshold=5.0):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    errors = np.abs(predictions - targets)
    correct = np.sum(errors < threshold)
    return correct / len(targets)

# -----------------------------
# 訓練模型
# -----------------------------
def train_model(air_quality_path, rainfall_path, weather_path, epochs=100, batch_size=2, lr=0.001, threshold=5.0):
    # 設定裝置 (GPU 或 CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用裝置: {device}")

    # 讀取資料
    X, y, scaler = load_and_preprocess_data(air_quality_path, rainfall_path, weather_path)

    # 拆分訓練集 / 測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 建立 Dataset 與 DataLoader
    train_dataset = AirQualityDataset(X_train, y_train)
    test_dataset = AirQualityDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 建立模型
    input_dim = X.shape[1]
    model = AirPollutionModel(input_dim)

    # 加入 LoRA 微調
    lora_config = LoraConfig(
        r=4,
        lora_alpha=15,
        target_modules=["fc1", "fc2"],
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    # 搬到 GPU
    model = model.to(device)

    # 定義 loss 與 optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 紀錄
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # 訓練迴圈
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        total_train_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)
            epoch_train_correct += calculate_accuracy(outputs.view(-1), targets, threshold) * inputs.size(0)
            total_train_samples += inputs.size(0)

        epoch_train_loss /= len(train_dataset)
        epoch_train_accuracy = epoch_train_correct / total_train_samples
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        # 驗證集
        model.eval()
        epoch_test_loss = 0.0
        epoch_test_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), targets)
                epoch_test_loss += loss.item() * inputs.size(0)
                epoch_test_correct += calculate_accuracy(outputs.view(-1), targets, threshold) * inputs.size(0)
                total_test_samples += inputs.size(0)

        epoch_test_loss /= len(test_dataset)
        epoch_test_accuracy = epoch_test_correct / total_test_samples
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_accuracy)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, "
                  f"Train Acc: {epoch_train_accuracy:.4f}, Test Acc: {epoch_test_accuracy:.4f}")

    # 保存結果與繪圖
    metrics_df = pd.DataFrame({
        'Epoch': range(1, epochs+1),
        'Train_Loss': train_losses,
        'Test_Loss': test_losses,
        'Train_Accuracy': train_accuracies,
        'Test_Accuracy': test_accuracies
    })
    metrics_df.to_csv(os.path.join('runs','metrics.csv'), index=False)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, epochs+1), test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.title('Loss'); plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(range(1, epochs+1), train_accuracies, label='Train Acc', color='blue')
    plt.plot(range(1, epochs+1), test_accuracies, label='Test Acc', color='red')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join('runs','metrics_curves.png'))
    plt.show()

    # 保存模型與 scaler
    torch.save(model.state_dict(), 'model.pth')
    joblib.dump(scaler, 'scaler.pkl')

    return model, scaler, train_losses, test_losses, train_accuracies, test_accuracies

# -----------------------------
# 主程式
# -----------------------------
if __name__ == "__main__":
    base_path = r'./data'
    air_quality_path = os.path.join(base_path,'air_quality_guting_combined_daily_average.csv')
    rainfall_path = os.path.join(base_path,'taipei_rainfall_combined.csv')
    weather_path = os.path.join(base_path,'taipei_weather_combined.csv')

    model, scaler, train_losses, test_losses, train_accuracies, test_accuracies = train_model(
        air_quality_path, rainfall_path, weather_path
    )
import torch
import torch.nn as nn

# 自訂 PyTorch 資料集
class AirQualityDataset(torch.utils.data.Dataset):
    """
    AirQualityDataset 用於封裝特徵 X 與目標值 y，
    方便 PyTorch DataLoader 讀取。
    """
    def __init__(self, X, y):
        """
        初始化資料集
        參數:
            X: 特徵的 numpy 陣列 (例如: 溫度、濕度、風速等)
            y: 目標值 (PM2.5) 的 numpy 陣列
        """
        self.X = torch.tensor(X, dtype=torch.float32)  # 轉成 float32 張量
        self.y = torch.tensor(y, dtype=torch.float32)  # 轉成 float32 張量
    
    def __len__(self):
        """
        回傳資料集的樣本數
        """
        return len(self.y)
    
    def __getitem__(self, idx):
        """
        透過索引 idx 取出一筆資料 (特徵, 目標值)
        """
        return self.X[idx], self.y[idx]

# 定義簡單的多層感知器 (MLP) 模型
class AirPollutionModel(nn.Module):
    """
    AirPollutionModel 是一個三層全連接的神經網路，用於預測 PM2.5
    """
    def __init__(self, input_dim):
        """
        初始化模型
        參數:
            input_dim: 特徵的維度 (X 的欄位數)
        """
        super(AirPollutionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 第一層，全連接層，輸入 -> 64 個神經元
        self.fc2 = nn.Linear(64, 32)         # 第二層，全連接層，64 -> 32
        self.fc3 = nn.Linear(32, 1)          # 輸出層，32 -> 1 (PM2.5 預測值)
        self.dropout = nn.Dropout(0.2)       # Dropout 層，防止過擬合，丟棄 20% 的神經元
    
    def forward(self, x):
        """
        前向傳播
        參數:
            x: 輸入特徵張量
        回傳:
            預測 PM2.5 的張量
        """
        x = torch.relu(self.fc1(x))  # 第一層使用 ReLU 激活
        x = self.dropout(x)           # 應用 dropout
        x = torch.relu(self.fc2(x))  # 第二層使用 ReLU 激活
        x = self.dropout(x)           # 再次應用 dropout
        x = self.fc3(x)               # 輸出層，不使用激活 (回歸任務)
        return x

# ================================
# 範例: 建立資料集與模型
# ================================
# 假設有 100 筆樣本，每筆 5 個特徵
# X = np.random.rand(100, 5)
# y = np.random.rand(100, 1)

# dataset = AirQualityDataset(X, y)
# model = AirPollutionModel(input_dim=5)
# print(model)

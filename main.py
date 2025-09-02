import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AirQualityDataset
from model import AirQualityTransformer
import os
import matplotlib.pyplot as plt

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    return running_loss / len(dataloader)

def main():
    # 設定參數
    data_dir = 'data'
    batch_size = 32
    num_epochs = 20
    seq_length = 24
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 資料載入
    train_dataset = AirQualityDataset(os.path.join(data_dir, 'train.csv'), seq_length=seq_length)
    val_dataset = AirQualityDataset(os.path.join(data_dir, 'val.csv'), seq_length=seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 模型、損失函數與優化器
    model = AirQualityTransformer(input_dim=12).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 訓練記錄
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    # 訓練迴圈
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'runs/best_model.pth')

    # 繪製訓練曲線
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('runs/training_curves.png')

if __name__ == '__main__':
    os.makedirs('runs', exist_ok=True)
    main()
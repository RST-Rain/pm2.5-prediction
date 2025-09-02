import torch
import torch.nn as nn

class AirQualityTransformer(nn.Module):
    def __init__(self, input_dim=12, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super(AirQualityTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)  # 將輸入特徵映射到 d_model 維度
        self.pos_encoder = nn.Parameter(torch.randn(1, 24, d_model))  # 位置編碼
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)  # 回歸輸出（PM2.5 濃度）

    def forward(self, x):
        x = self.input_fc(x)  # [batch, seq_length, input_dim] -> [batch, seq_length, d_model]
        x = x + self.pos_encoder  # 加入位置編碼
        x = self.transformer(x)   # Transformer 處理
        x = x[:, -1, :]           # 取最後一個時間步
        x = self.fc_out(x)        # 回歸預測
        return x.squeeze(-1)      # [batch]
# src/model_def.py

import torch
import torch.nn as nn

class MLPRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64):
        super(MLPRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # 测试模型定义是否有效
    input_dim = 10  # 假设有10个特征
    model = MLPRegression(input_dim, hidden_dim1=128, hidden_dim2=64)
    print("Model Architecture:")
    print(model)

    # 测试前向传播
    dummy_input = torch.randn(32, input_dim)  # 32个样本
    output = model(dummy_input)
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Dummy output shape: {output.shape}")






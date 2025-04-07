import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn


# 模型构建
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 模型预测与结果反归一化
def predict_future(model, X_test, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    X_test = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        predictions = model(X_test)
        predictions = predictions.cpu().numpy()

    # 反归一化，需要构建一个全零数组来辅助恢复收盘价
    num_predictions = predictions.shape[0] * predictions.shape[1]
    dummy_array = np.zeros((num_predictions, scaler.n_features_in_))
    flat_predictions = predictions.flatten()
    for i in range(num_predictions):
        dummy_array[i, 3] = flat_predictions[i]
    predictions = scaler.inverse_transform(dummy_array)[:, 3].reshape(predictions.shape)

    return predictions


# 推理函数
def infer(model_path, scaler_path, file_path, seq_length, future_steps):
    # 加载 scaler
    scaler = MinMaxScaler()
    scaler.fit(np.load(scaler_path))

    # 加载并预处理数据
    data = pd.read_csv(file_path)
    features = data[['open', 'high', 'low', 'close', 'volume']].values
    scaled_features = scaler.transform(features)

    # 初始化模型
    input_size = 5
    hidden_size = 64
    num_layers = 2
    output_size = future_steps
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    # 设置 weights_only=True
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # 准备预测数据
    last_sequence = scaled_features[:seq_length].reshape(1, seq_length, input_size)
    future_predictions = predict_future(model, last_sequence, scaler)

    # 获取真实结果
    real_close_prices = data['close'].values
    real_future_prices = real_close_prices[366:372]

    # 输出预测结果和真实结果的对照
    print("预测结果:", future_predictions[0])
    print("真实结果:", real_future_prices)


# 1. 数据读取与预处理
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # 选取需要的特征列
    features = data[['open', 'high', 'low', 'close', 'volume']].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler


if __name__ == "__main__":
    file_path = '/home/ljh/StockPredictor/bj430139_daily_data.csv'
    seq_length = 365  # 选取365天数据作为一个样本
    future_steps = 6  # 预测第367天到372天，共6天
    # 这里需要手动指定训练生成的模型名称和 scaler 文件名称
    model_name = '/home/ljh/StockPredictor/bj430139_daily_data_2022-10-28_2024-04-29_20250403.pth'
    scaler_path = f'{model_name}_scaler.npy'

    infer(model_name, scaler_path, file_path, seq_length, future_steps)
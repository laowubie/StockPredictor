# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
#
# # 1. 数据读取与预处理
# def load_and_preprocess_data(file_path):
#     data = pd.read_csv(file_path)
#     # 选取需要的特征列
#     features = data[['open', 'high', 'low', 'close', 'volume']].values
#     scaler = MinMaxScaler()
#     scaled_features = scaler.fit_transform(features)
#     return scaled_features, scaler
#
#
# # 2. 数据特征工程
# def create_sequences(data, seq_length, future_steps):
#     X, y = [], []
#     for i in range(len(data) - seq_length - future_steps + 1):
#         X.append(data[i:i + seq_length])
#         # 只取收盘价作为预测目标
#         y.append(data[i + seq_length:i + seq_length + future_steps, 3])
#
#     X = np.array(X)
#     y = np.array(y)
#
#     return X, y
#
#
# # 3. 模型构建
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out
#
#
# # 4. 模型训练
# def train_model(model, X_train, y_train, epochs, batch_size, learning_rate):
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#
#     X_train = torch.FloatTensor(X_train).to(device)
#     y_train = torch.FloatTensor(y_train).to(device)
#
#     for epoch in range(epochs):
#         model.train()
#         for i in range(0, len(X_train), batch_size):
#             inputs = X_train[i:i + batch_size]
#             targets = y_train[i:i + batch_size]
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#
#         if (epoch + 1) % 10 == 0:
#             print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
#
#
# # 5. 模型预测与结果反归一化
# def predict_future(model, X_test, scaler):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     X_test = torch.FloatTensor(X_test).to(device)
#
#     with torch.no_grad():
#         predictions = model(X_test)
#         predictions = predictions.cpu().numpy()
#
#     # 反归一化，需要构建一个全零数组来辅助恢复收盘价
#     num_predictions = predictions.shape[0] * predictions.shape[1]
#     dummy_array = np.zeros((num_predictions, scaler.n_features_in_))
#     flat_predictions = predictions.flatten()
#     for i in range(num_predictions):
#         dummy_array[i, 3] = flat_predictions[i]
#     predictions = scaler.inverse_transform(dummy_array)[:, 3].reshape(predictions.shape)
#
#     return predictions
#
#
# # 主程序
# if __name__ == "__main__":
#     file_path = '/home/ljh/StockPredictor/bj430139_daily_data.csv'
#     seq_length = 30  # 过去30天数据作为一个样本
#     future_steps = 5  # 预测未来5天
#
#     # 加载并预处理数据
#     scaled_features, scaler = load_and_preprocess_data(file_path)
#
#     # 创建训练序列
#     X, y = create_sequences(scaled_features, seq_length, future_steps)
#
#     # 划分训练集和测试集（简单示例，可按需调整）
#     train_size = int(len(X) * 0.8)
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]
#
#     # 初始化模型
#     input_size = 5  # 因为使用了5个特征
#     hidden_size = 64
#     num_layers = 2
#     output_size = future_steps
#     model = LSTMModel(input_size, hidden_size, num_layers, output_size)
#
#     # 训练模型
#     epochs = 50
#     batch_size = 32
#     learning_rate = 0.001
#     train_model(model, X_train, y_train, epochs, batch_size, learning_rate)
#
#     # 预测未来五天
#     last_sequence = scaled_features[-seq_length:].reshape(1, seq_length, input_size)
#     future_predictions = predict_future(model, last_sequence, scaler)
#     print("未来五天的预测收盘价:", future_predictions[0])




















import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim


# 1. 数据读取与预处理
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # 选取需要的特征列
    features = data[['open', 'high', 'low', 'close', 'volume']].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler


# 2. 数据特征工程
def create_sequences(data, seq_length, future_steps):
    X, y = [], []
    for i in range(len(data) - seq_length - future_steps + 1):
        X.append(data[i:i + seq_length])
        # 只取收盘价作为预测目标
        y.append(data[i + seq_length:i + seq_length + future_steps, 3])

    X = np.array(X)
    y = np.array(y)

    return X, y


# 3. 模型构建
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


# 4. 模型训练
def train_model(model, X_train, y_train, epochs, batch_size, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i + batch_size]
            targets = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


# 5. 模型预测与结果反归一化
def predict_future(model, X_test, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


# 主程序
if __name__ == "__main__":
    file_path = '/home/ljh/StockPredictor/bj430139_daily_data.csv'
    seq_length = 90  # 选取365天数据作为一个样本
    future_steps = 5  # 预测第367天到372天，共6天

    # 加载并预处理数据
    scaled_features, scaler = load_and_preprocess_data(file_path)

    # 创建训练序列
    X, y = create_sequences(scaled_features, seq_length, future_steps)

    # 选取前365天数据进行训练
    X_train = X[:1]
    y_train = y[:1]

    # 初始化模型
    input_size = 5  # 因为使用了5个特征
    hidden_size = 64
    num_layers = 2
    output_size = future_steps
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # 训练模型
    epochs = 50
    batch_size = 1
    learning_rate = 0.001
    train_model(model, X_train, y_train, epochs, batch_size, learning_rate)

    # 准备预测数据
    last_sequence = scaled_features[:seq_length].reshape(1, seq_length, input_size)
    future_predictions = predict_future(model, last_sequence, scaler)

    # 获取真实结果
    real_data = pd.read_csv(file_path)
    real_close_prices = real_data['close'].values
    real_future_prices = real_close_prices[366:372]

    # 输出预测结果和真实结果的对照
    print("预测结果:", future_predictions[0])
    print("真实结果:", real_future_prices)
























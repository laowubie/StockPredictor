# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from datetime import datetime
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
#     return model
#
#
# # 训练函数
# def train(file_path, seq_length, future_steps, epochs, batch_size, learning_rate):
#     # 加载并预处理数据
#     scaled_features, scaler = load_and_preprocess_data(file_path)
#
#     # 创建训练序列
#     X, y = create_sequences(scaled_features, seq_length, future_steps)
#
#     # 选取前365天数据进行训练
#     X_train = X[:1]
#     y_train = y[:1]
#
#     # 初始化模型
#     input_size = 5  # 因为使用了5个特征
#     hidden_size = 64
#     num_layers = 2
#     output_size = future_steps
#     model = LSTMModel(input_size, hidden_size, num_layers, output_size)
#
#     # 训练模型
#     trained_model = train_model(model, X_train, y_train, epochs, batch_size, learning_rate)
#
#     # 获取数据文件名和日期范围
#     data_name = file_path.split('/')[-1].split('.')[0]
#     data = pd.read_csv(file_path)
#     start_date = data['date'].iloc[0]
#     end_date = data['date'].iloc[seq_length - 1]
#
#     # 生成模型保存名称
#     current_date = datetime.now().strftime("%Y%m%d")
#     model_name = f"{data_name}_{start_date}_{end_date}_{current_date}.pth"
#
#     # 保存模型
#     torch.save(trained_model.state_dict(), model_name)
#     print(f"模型已保存到 {model_name}")
#     return model_name, scaler
#
#
# if __name__ == "__main__":
#     file_path = '/home/ljh/StockPredictor/bj430139_daily_data.csv'
#     seq_length = 365  # 选取365天数据作为一个样本
#     future_steps = 6  # 预测第367天到372天，共6天
#     epochs = 50
#     batch_size = 1
#     learning_rate = 0.001
#
#     # 训练
#     model_name, scaler = train(file_path, seq_length, future_steps, epochs, batch_size, learning_rate)
#     # 保存 scaler
#     np.save(f'{model_name}_scaler.npy', scaler.transform(np.eye(5)))
#









import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime


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
    return model


# 训练函数
def train(file_path, seq_length, future_steps, epochs, batch_size, learning_rate):
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
    trained_model = train_model(model, X_train, y_train, epochs, batch_size, learning_rate)

    # 获取数据文件名和日期范围
    data_name = file_path.split('/')[-1].split('.')[0]
    data = pd.read_csv(file_path)
    start_date = data['date'].iloc[0]
    end_date = data['date'].iloc[seq_length - 1]

    # 生成模型保存名称
    current_date = datetime.now().strftime("%Y%m%d")
    model_name = f"{data_name}_{start_date}_{end_date}_{current_date}.pth"

    # 保存模型
    torch.save(trained_model.state_dict(), model_name)
    print(f"模型已保存到 {model_name}")
    # 保存 scaler
    np.save(f'{model_name}_scaler.npy', scaler.transform(np.eye(5)))
    return model_name, scaler


if __name__ == "__main__":
    file_path = '/home/ljh/StockPredictor/bj430139_daily_data.csv'
    seq_length = 365  # 选取365天数据作为一个样本
    future_steps = 6  # 预测第367天到372天，共6天
    epochs = 50
    batch_size = 1
    learning_rate = 0.001

    # 训练
    model_name, scaler = train(file_path, seq_length, future_steps, epochs, batch_size, learning_rate)
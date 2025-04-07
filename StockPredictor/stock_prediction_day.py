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
        loss = None  # 初始化 loss 变量
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i + batch_size]
            targets = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if loss is not None and (epoch + 1) % 10 == 0:
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
    file_path = '/home/ljh/StockPredictor/bj430017_daily_data.csv'
    seq_length = 30  # 选取365天数据作为一个样本
    future_steps = 5  # 预测第367天到372天，共6天
    end_date = '2025-01-03'  # 可修改为你想要的结束日期

    # 加载数据
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])  # 假设数据中有'date'列

    # 检查是否存在符合 end_date 的数据
    end_date_rows = data[data['date'] == end_date]
    if end_date_rows.empty:
        print(f"错误：数据中未找到日期为 {end_date} 的记录，请检查日期设置。")
        import sys

        sys.exit(1)

    end_index = end_date_rows.index[0]
    # 按顺序选取 seq_length 个交易日的数据
    selected_indices = []
    current_index = end_index
    count = 0
    while count < seq_length and current_index >= 0:
        selected_indices.append(current_index)
        current_index -= 1
        count += 1

    selected_indices.sort()

    # 加载并预处理数据
    scaled_features, scaler = load_and_preprocess_data(file_path)

    # 创建训练序列
    X, y = create_sequences(scaled_features, seq_length, future_steps)

    # 选取指定日期范围内的数据进行训练
    X_train = X[selected_indices[-1]:selected_indices[-1] + 1]
    y_train = y[selected_indices[-1]:selected_indices[-1] + 1]

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
    last_sequence = scaled_features[selected_indices[0]:selected_indices[-1] + 1].reshape(1, seq_length, input_size)
    future_predictions = predict_future(model, last_sequence, scaler)

    # 获取真实结果
    real_close_prices = data['close'].values
    real_future_prices = real_close_prices[end_index + 1:end_index + 1 + future_steps]

    # 获取预测数据的日期
    prediction_dates = data['date'][end_index + 1:end_index + 1 + future_steps].tolist()

    # 规范输出排版，将预测和真实结果对照输出
    print("-" * 70)
    print(f"| {'日期':^12} | {'预测收盘价':^15} | {'真实收盘价':^15} |")
    print("-" * 70)
    for date, prediction, real_price in zip(prediction_dates, future_predictions[0], real_future_prices):
        formatted_prediction = f"{prediction:.4f}"
        formatted_real_price = f"{real_price:.4f}"
        print(f"| {date.strftime('%Y-%m-%d'):^12} | {formatted_prediction:^15} | {formatted_real_price:^15} |")
    print("-" * 70)

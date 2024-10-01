import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from copy import deepcopy as dc
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

# تنظیم دانه تصادفی
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Data loading
directory_path = '../100stocks'  # مسیر پوشه داده‌های شما
dataframes = {}

for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        stock_name = filename.split('.')[0]
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        dataframes[stock_name] = df

# Sliding Window Creation
def create_time_based_sliding_window(dataFrame, n_steps, stock_name, start_date, date_step='1D'):
    df = dc(dataFrame[stock_name])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Adj Close']]
    df.set_index('Date', inplace=True)
    df = df.resample(date_step).last().dropna()

    # Create shifted columns (time windows) for the target stock
    shifted_columns = [df['Adj Close'].shift(i).rename(f'Adj Close(t-{i})') for i in range(1, n_steps + 1)]
    df = pd.concat([df] + shifted_columns, axis=1).dropna()

    # Convert start_date to datetime and filter target stock data
    start_date = pd.to_datetime(start_date)
    df = df[df.index >= start_date]
    
    return df

# Prepare Data for training
def create_sequences(data):
    xs, ys = [], []
    for i in range(len(data)):
        x = data[i, 1:]  # Features
        y = data[i, 0]  # Target (Adj Close)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Reshape data for LSTM input
def reshape_data(X, y, lookback):
    num_features = X.shape[1]
    num_samples = (len(X) - lookback)

    X_reshaped = np.zeros((num_samples, lookback, num_features))
    y_reshaped = np.zeros((num_samples, 1))

    for i in range(num_samples):
        X_reshaped[i] = X[i:i + lookback]
        y_reshaped[i] = y[i + lookback - 1]

    return X_reshaped, y_reshaped

# LSTM Model Definition
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Dataset and DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# Training and Validation Functions
def train_one_epoch(epoch, model, train_loader, optimizer, loss_function, device):
    model.train()
    running_loss = 0.0
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}] Total Loss: {running_loss / len(train_loader):.6f}')

def validate_one_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
    avg_val_loss = running_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.6f}')
    return avg_val_loss

# Main Loop for Processing Stocks
errors = []
lookback = 270  # تنظیم مقدار lookback

# ایجاد فولدر برای qaz
lookback_dir = f'lookback_{lookback}'
plot_dir = os.path.join(lookback_dir, 'stock_plots') 
os.makedirs(plot_dir, exist_ok=True)

# فایل ارورها
error_file = os.path.join(lookback_dir, f'stock_errors_lookback_{lookback}.csv')

# پارامترهای مدل
batch_size = 32
hidden_size = 200
num_layers = 3
dropout_rate = 0.2
learning_rate = 0.0001
num_epochs = 100
patience = 20
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_function = nn.MSELoss()

for stock_name in dataframes.keys():
    print(f"Processing stock: {stock_name}")
    
    try:
        # بررسی اینکه سهام داده‌ای در سال ۲۰۱۵ دارد یا خیر
        df = dc(dataframes[stock_name])
        df['Date'] = pd.to_datetime(df['Date'])
        df_2015 = df[(df['Date'] >= '2015-01-01') & (df['Date'] < '2016-01-01')]

        if len(df_2015) == 0:
            print(f"Skipping {stock_name}: No data available in 2015.")
            continue  # رد سهام‌هایی که در ۲۰۱۵ داده‌ای ندارند

        # آماده‌سازی داده‌ها
        shifted_df = create_time_based_sliding_window(dataframes, lookback, stock_name, start_date='2015-01-01', date_step='1D')

        if len(shifted_df) == 0:
            print(f"No data available for stock {stock_name} after the start date.")
            continue

        shifted_df_as_np = shifted_df.to_numpy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(shifted_df_as_np)

        # ایجاد توالی‌ها
        X, y = create_sequences(scaled_data)

        # تقسیم داده‌ها به مجموعه‌های آموزشی، اعتبارسنجی و تست
        train_index = int(len(X) * train_ratio)
        validation_index = int(len(X) * (train_ratio + validation_ratio))

        X_train, y_train = X[:train_index], y[:train_index]
        X_val, y_val = X[train_index:validation_index], y[train_index:validation_index]
        X_test, y_test = X[validation_index:], y[validation_index:]

        # شکل‌دهی دوباره داده‌ها
        X_train, y_train = reshape_data(X_train, y_train, lookback)
        X_val, y_val = reshape_data(X_val, y_val, lookback)
        X_test, y_test = reshape_data(X_test, y_test, lookback)

        # تبدیل داده‌ها به تنسورهای PyTorch
        X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
        X_val, y_val = torch.tensor(X_val).float(), torch.tensor(y_val).float()
        X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

        # ایجاد DataLoader
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # ایجاد مدل LSTM
        input_size = X_train.shape[2]
        model = LSTM(input_size, hidden_size, num_layers, dropout_rate).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            train_one_epoch(epoch, model, train_loader, optimizer, loss_function, device)
            val_loss = validate_one_epoch(model, val_loader, loss_function, device)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = dc(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered for {stock_name}")
                    break

        model.load_state_dict(best_model_wts)

        # پیش‌بینی و محاسبه متریک‌ها
        with torch.no_grad():
            test_predictions = model(X_test.to(device)).cpu().numpy().flatten()

        mse = mean_squared_error(y_test.cpu().numpy().flatten(), test_predictions)
        mae = mean_absolute_error(y_test.cpu().numpy().flatten(), test_predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test.cpu().numpy().flatten(), test_predictions)

        # ذخیره ارورها
        errors.append([stock_name, mse, mae, rmse, mape])
        errors_df = pd.DataFrame(errors, columns=['Stock', 'MSE', 'MAE', 'RMSE', 'MAPE'])
        errors_df.to_csv(error_file, index=False)

        # بازگردانی پیش‌بینی‌ها به مقیاس اصلی
        dummies_test = np.zeros((X_test.shape[0], input_size + 1))
        dummies_test[:, 0] = test_predictions
        dummies_test = scaler.inverse_transform(dummies_test)
        test_predictions_unscaled = dummies_test[:, 0]

        dummies_actual_test = np.zeros((X_test.shape[0], input_size + 1))
        dummies_actual_test[:, 0] = y_test.cpu().numpy().flatten()
        dummies_actual_test = scaler.inverse_transform(dummies_actual_test)
        new_y_test = dummies_actual_test[:, 0]

        # ذخیره نمودار پیش‌بینی و واقعی
        plt.figure(figsize=(10, 6))
        plt.plot(new_y_test, label='Actual Close', color='blue')
        plt.plot(test_predictions_unscaled, label='Predicted Close', color='orange')
        plt.xlabel('Day')
        plt.ylabel('Close Price')
        plt.title(f'Actual vs Predicted Close Price (Test Set) for {stock_name}')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{stock_name}_lookback_{lookback}_actual_vs_predicted_close_test.png'))

        print(f"Completed processing for {stock_name}")

    except Exception as e:
        print(f"Error processing stock {stock_name}: {str(e)}")

print("All stocks processed.")

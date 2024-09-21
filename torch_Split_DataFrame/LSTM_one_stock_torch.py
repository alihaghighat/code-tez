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

set_seed(42)  # می‌توانید مقدار دانه را تغییر دهید

# Data loading
directory_path = '../100stocks'
dataframes = {}

for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        stock_name = filename.split('.')[0]
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        dataframes[stock_name] = df

# Sliding Window Creation
def create_time_based_sliding_window(dataFrame, n_steps, stock_name, date_step='1D'):
    df = dc(dataFrame[stock_name])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Adj Close']]
    df.set_index('Date', inplace=True)
    df = df.resample(date_step).last().dropna()
    shifted_columns = [df['Adj Close'].shift(i).rename(f'Adj Close(t-{i})') for i in range(1, n_steps + 1)]
    df = pd.concat([df] + shifted_columns, axis=1)
    df.dropna(inplace=True)
    return df

# Prepare Data
lookback = 180
shifted_aapl_df = create_time_based_sliding_window(dataframes, lookback, 'AAPL', '1D')
shifted_df_as_np = shifted_aapl_df.to_numpy()
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]
X = dc(np.flip(X, axis=1))

# Train/Validation/Test Split
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15
train_index = int(len(X) * train_ratio)
validation_index = int(len(X) * (train_ratio + validation_ratio))

X_train = X[:train_index]
X_val = X[train_index:validation_index]
X_test = X[validation_index:]
y_train = y[:train_index]
y_val = y[train_index:validation_index]
y_test = y[validation_index:]

# Reshape for LSTM
X_train = X_train.reshape((-1, lookback, 1))
X_val = X_val.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))
y_train = y_train.reshape((-1, 1))
y_val = y_val.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# Convert to torch tensors
X_train = torch.tensor(X_train).float()
X_val = torch.tensor(X_val).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).float()
y_val = torch.tensor(y_val).float()
y_test = torch.tensor(y_test).float()

# Dataset and DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

batch_size = 32
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        out = self.fc(out[:, -1, :])  # Only take the output of the last time step
        return out

input_size = 1
hidden_size = 200
num_layers = 3
dropout_rate = 0.2
model = LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate).to(device)

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
        if (batch_index + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}], Batch [{batch_index + 1}], Loss: {running_loss / (batch_index + 1):.6f}')
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
    return avg_val_loss  # اطمینان از بازگشت مقدار

# Training Loop
learning_rate = 0.0001
num_epochs = 100
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # افزودن Weight Decay

# تعریف Scheduler بدون پارامتر verbose
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Early Stopping parameters
best_val_loss = float('inf')
patience = 20
counter = 0

for epoch in range(num_epochs):
    train_one_epoch(epoch, model, train_loader, optimizer, loss_function, device)
    val_loss = validate_one_epoch(model, val_loader, loss_function, device)
    
    # افزودن دستور چاپ برای دیباگ
    print(f'Epoch {epoch+1}: val_loss={val_loss}, type={type(val_loss)}')
    
    if val_loss is None:
        raise ValueError("Validation loss is None. بررسی کنید که تابع validate_one_epoch به درستی مقدار باز می‌گرداند.")
    
    # به‌روزرسانی Scheduler با مقدار val_loss
    scheduler.step(val_loss)
    
    # مشاهده نرخ یادگیری فعلی
    current_lr = scheduler.get_last_lr()
    print(f'Current learning rate: {current_lr}')
    
    # بررسی بهترین مقدار val_loss برای Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = dc(model.state_dict())
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# بارگذاری بهترین وزن‌ها
model.load_state_dict(best_model_wts)

# ادامه کد برای پیش‌بینی و ارزیابی مدل
with torch.no_grad():
    predicted = model(X_train.to(device)).cpu().numpy()

y_train_cpu = y_train.cpu().numpy()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(y_train_cpu, label='Actual Close')
plt.plot(predicted, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Price')
plt.legend()
plt.savefig('actual_vs_predicted_close.png')

# Unscale predictions and actual values
train_predictions = predicted.flatten()
dummies_train = np.zeros((X_train.shape[0], lookback + 1))
dummies_predictions = np.zeros((X_train.shape[0], lookback + 1))
dummies_train[:, 0] = y_train.flatten()
dummies_predictions[:, 0] = train_predictions

dummies_train = scaler.inverse_transform(dummies_train)
dummies_predictions = scaler.inverse_transform(dummies_predictions)

new_y_train = dummies_train[:, 0]
train_predictions_unscaled = dummies_predictions[:, 0]

plt.figure(figsize=(10, 6))
plt.plot(new_y_train, label='Actual Close', color='blue')
plt.plot(train_predictions_unscaled, label='Predicted Close', color='orange')
plt.xlabel('Day')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Price (Unscaled)')
plt.legend()
plt.savefig('actual_vs_predicted_close_unscaled.png')

# Test Set Predictions and Metrics
with torch.no_grad():
    test_predictions = model(X_test.to(device)).cpu().numpy().flatten()

# محاسبه متریک‌ها
mse = mean_squared_error(y_test.cpu().numpy().flatten(), test_predictions)
mae = mean_absolute_error(y_test.cpu().numpy().flatten(), test_predictions)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test.cpu().numpy().flatten(), test_predictions)

print(f"Mean Squared Error (MSE) on scaled data: {mse:.6f}")
print(f"Mean Absolute Error (MAE) on scaled data: {mae:.6f}")
print(f"Root Mean Squared Error (RMSE) on scaled data: {rmse:.6f}")
print(f"Mean Absolute Percentage Error (MAPE) on scaled data: {mape:.2f}%")

# Inverse transform test predictions
dummies_test = np.zeros((X_test.shape[0], lookback + 1))
dummies_test[:, 0] = test_predictions
dummies_test = scaler.inverse_transform(dummies_test)

test_predictions_unscaled = dummies_test[:, 0]

# Inverse transform y_test
dummies_actual_test = np.zeros((X_test.shape[0], lookback + 1))
dummies_actual_test[:, 0] = y_test.cpu().numpy().flatten()
dummies_actual_test = scaler.inverse_transform(dummies_actual_test)

new_y_test = dummies_actual_test[:, 0]

plt.figure(figsize=(10, 6))
plt.plot(new_y_test, label='Actual Close', color='blue')
plt.plot(test_predictions_unscaled, label='Predicted Close', color='orange')
plt.xlabel('Day')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Price (Test Set)')
plt.legend()
plt.savefig('actual_vs_predicted_close_test.png')

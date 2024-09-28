import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
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

def pearson_correlation(x, y):
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    deviation_product = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = (sum((xi - mean_x) ** 2 for xi in x) / len(x)) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y) / len(y)) ** 0.5
    correlation = deviation_product / (len(x) * std_x * std_y)
    return correlation
def calculate_lagged_correlation(stock_data_1, stock_data_2, start_date, window_size, lag):
    # Filter data starting from the specified date
    stock_data_1_filtered = stock_data_1[stock_data_1['Date'] <= start_date].reset_index(drop=True)
    stock_data_2_filtered = stock_data_2[stock_data_2['Date'] <= start_date].reset_index(drop=True)

    if len(stock_data_1_filtered) == 0 or len(stock_data_2_filtered) == 0:
        return -1,None

    # Ensure both dataframes have the same length after filtering
    min_length = min(len(stock_data_1_filtered), len(stock_data_2_filtered))
    stock_data_1_filtered = stock_data_1_filtered.iloc[:min_length]
    stock_data_2_filtered = stock_data_2_filtered.iloc[:min_length]
    
    
    if lag > 0:
        # Check if both filtered datasets have enough data for the window and lag
        if len(stock_data_1_filtered) < window_size + lag or len(stock_data_2_filtered) < window_size:
            return -1,None
        
        # Extract windows with dates included
        window_1_with_dates = stock_data_1_filtered[['Date', 'Adj Close']].iloc[-window_size - lag:-lag].reset_index(drop=True)
        window_2_with_dates = stock_data_2_filtered[['Date', 'Adj Close']].iloc[-window_size:].reset_index(drop=True)
        
        # # Print windows with dates
        # print("Window 1 with dates:\n", window_1_with_dates)
        # print("Window 2 with dates:\n", window_2_with_dates)
        
        # Remove dates for correlation calculation
        window_1 = window_1_with_dates['Adj Close']
        window_2 = window_2_with_dates['Adj Close']

    elif lag < 0:
        # Check if both filtered datasets have enough data for the window and lag
        if len(stock_data_2_filtered) < window_size + abs(lag) or len(stock_data_1_filtered) < window_size:
            return -1,None

        # Extract windows with dates included
        window_1_with_dates = stock_data_1_filtered[['Date', 'Adj Close']].iloc[-window_size:].reset_index(drop=True)
        window_2_with_dates = stock_data_2_filtered[['Date', 'Adj Close']].iloc[-window_size - abs(lag):-abs(lag)].reset_index(drop=True)
        
        # # Print windows with dates
        # print("Window 1 with dates:\n", window_1_with_dates)
        # print("Window 2 with dates:\n", window_2_with_dates)
        
        # Remove dates for correlation calculation
        window_1 = window_1_with_dates['Adj Close']
        window_2 = window_2_with_dates['Adj Close']

    else:
        # No lag, use data as is
        if len(stock_data_1_filtered) < window_size or len(stock_data_2_filtered) < window_size:
             return -1,None

        # Extract windows with dates included
        window_1_with_dates = stock_data_1_filtered[['Date', 'Adj Close']].iloc[-window_size:].reset_index(drop=True)
        window_2_with_dates = stock_data_2_filtered[['Date', 'Adj Close']].iloc[-window_size:].reset_index(drop=True)
        
        # # Print windows with dates
        # print("Window 1 with dates:\n", window_1_with_dates)
        # print("Window 2 with dates:\n", window_2_with_dates)
        
        # Remove dates for correlation calculation
        window_1 = window_1_with_dates['Adj Close']
        window_2 = window_2_with_dates['Adj Close']

    # Check if the windows are valid and have the same length
    if len(window_1) != len(window_2):
        return -1,None

    # Calculate Pearson correlation
    correlation = pearson_correlation(window_1, window_2)

    return correlation,window_2_with_dates
def create_time_based_sliding_window(dataFrame, n_steps, stock_name, date_step='1D', n_top=3, delay=1, start_date='2018-01-01'):
    # Copy the specific stock data and resample
    df = dc(dataFrame[stock_name])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Adj Close']]
    df.set_index('Date', inplace=True)
    df = df.resample(date_step).last().dropna()

    # Create shifted columns (time windows) for the target stock
    shifted_columns = [df['Adj Close'].shift(i).rename(f'Adj Close(t-{i})') for i in range(1, n_steps + 1)]
    df = pd.concat([df] + shifted_columns, axis=1).dropna()

    # Prepare the other stocks data in advance
    other_stocks_data = {col: dataFrame[col].copy() for col in dataFrame.keys() if col != stock_name}
    for stock, stock_df in other_stocks_data.items():
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        other_stocks_data[stock] = stock_df

    # Convert start_date to datetime and filter target stock data
    start_date = pd.to_datetime(start_date)
    df = df[df.index >= start_date]

    # Create a dictionary to store new columns for each row
    all_new_columns = {}

    # Iterate over rows from the start_date onward
    for row_index in df.index:
        print(f"Processing row: {row_index}")
        correlations = []
        
        # Compare each stock with the target stock
        for other_stock, other_stock_data in other_stocks_data.items():
            date_str = row_index.strftime('%Y-%m-%d')
            
            # Calculate correlation
            correlation, window_2_with_dates = calculate_lagged_correlation(
                dataFrame[stock_name],
                other_stock_data,
                date_str,
                n_steps,
                delay
            )
            
            correlations.append((other_stock, correlation, window_2_with_dates))

        # Sort correlations and keep the top n
        correlations = [(stock, corr, window_2) for stock, corr, window_2 in correlations if corr > 0]
        if correlations:
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            top_correlated_stocks = correlations[:n_top]
            
            # Create a dictionary to store new columns for this row
            new_columns = {f'Top_{i+1} Close(t-{j+1})': value
                           for i, (_, _, window_data) in enumerate(top_correlated_stocks)
                           for j, value in enumerate(window_data['Adj Close'].to_numpy()[::-1])}
            
            # Store the new columns for the current row in the global dictionary
            all_new_columns[row_index] = new_columns
        
    # After processing all rows, create a DataFrame from the dictionary
    new_columns_df = pd.DataFrame.from_dict(all_new_columns, orient='index')
      # Drop the columns Adj Close(t-1) to Adj Close(t-30)
    columns_to_drop = [f'Adj Close(t-{i})' for i in range(1, n_steps + 1)]
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    # Concatenate the original DataFrame with the new columns at once
    df = pd.concat([df, new_columns_df], axis=1)

    return df





# Prepare Data
lookback = 30
# نام فایل CSV که می‌خواهید داده‌ها را در آن ذخیره کنید
csv_file = 'shifted_aapl_df'+str(lookback)+'.csv'

# بررسی اینکه آیا فایل CSV وجود دارد یا خیر
if os.path.exists(csv_file):
    # اگر فایل وجود دارد، آن را بارگذاری کنید
    shifted_aapl_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    print("Data loaded from CSV file.")
else:
    # اگر فایل وجود ندارد، داده‌ها را بسازید و ذخیره کنید
    shifted_aapl_df = create_time_based_sliding_window(dataframes, lookback, 'AAPL', '1D', 3, -2, start_date='2016-01-01')
    # ذخیره‌سازی داده‌ها در فایل CSV
    shifted_aapl_df.to_csv(csv_file)
    print("Data created and saved to CSV file.")

# نمایش DataFrame
print(shifted_aapl_df)
shifted_df_as_np = shifted_aapl_df.to_numpy()

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(shifted_df_as_np)
def create_sequences(data):
    xs, ys = [], []
    for i in range(len(data)):
        x = data[i,1:]  # Features
        y = data[i , 0]  # Target (Adj Close)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(scaled_data)
print(X.shape)
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15
train_index = int(len(X) * train_ratio)
validation_index = int(len(X) * (train_ratio + validation_ratio))


def reshape_data(X, y, lookback):
    # Automatically get the number of features from X
    num_features = X.shape[1]  # Assuming X is 2D (n_samples, n_features)
    
    # Calculate the number of samples
    num_samples = (len(X) - lookback)  # Overlapping windows for prediction

    # Initialize reshaped arrays
    X_reshaped = np.zeros((num_samples, lookback, num_features))
    y_reshaped = np.zeros((num_samples, 1))

    for i in range(num_samples):
        X_reshaped[i] = X[i:i + lookback]  # Get the window of lookback days
        y_reshaped[i] = y[i + lookback - 1]  # Get the corresponding label for the next day

    return X_reshaped, y_reshaped

# Example usage with your indices
X_train = X[:train_index]
X_val = X[train_index:validation_index]
X_test = X[validation_index:]

y_train = y[:train_index]
y_val = y[train_index:validation_index]
y_test = y[validation_index:]

# Reshape data
X_train, y_train = reshape_data(X_train, y_train, lookback)
X_val, y_val = reshape_data(X_val, y_val, lookback)
X_test, y_test = reshape_data(X_test, y_test, lookback)

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

input_size = X_train.shape[2] 
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

# Get the number of features from the input
n_features = X_train.shape[2]  # Assuming X_train is shaped (n_samples, lookback, n_features)

# Prepare dummies for actual and predicted values
dummies_train = np.zeros((X_train.shape[0], n_features + 1))  # +1 for the actual value
dummies_predictions = np.zeros((X_train.shape[0], n_features + 1))

# Fill the first column with actual values and predictions
dummies_train[:, 0] = y_train.flatten()
dummies_predictions[:, 0] = train_predictions

# Inverse transform to get unscaled values
dummies_train = scaler.inverse_transform(dummies_train)
dummies_predictions = scaler.inverse_transform(dummies_predictions)

# Extract the unscaled actual and predicted values
new_y_train = dummies_train[:, 0]
train_predictions_unscaled = dummies_predictions[:, 0]

# Plot for training set
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

# Calculate metrics
mse = mean_squared_error(y_test.cpu().numpy().flatten(), test_predictions)
mae = mean_absolute_error(y_test.cpu().numpy().flatten(), test_predictions)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test.cpu().numpy().flatten(), test_predictions)

print(f"Mean Squared Error (MSE) on scaled data: {mse:.6f}")
print(f"Mean Absolute Error (MAE) on scaled data: {mae:.6f}")
print(f"Root Mean Squared Error (RMSE) on scaled data: {rmse:.6f}")
print(f"Mean Absolute Percentage Error (MAPE) on scaled data: {mape:.2f}%")

# Inverse transform test predictions
dummies_test = np.zeros((X_test.shape[0], n_features + 1))
dummies_test[:, 0] = test_predictions
dummies_test = scaler.inverse_transform(dummies_test)

test_predictions_unscaled = dummies_test[:, 0]

# Inverse transform y_test
dummies_actual_test = np.zeros((X_test.shape[0], n_features + 1))
dummies_actual_test[:, 0] = y_test.cpu().numpy().flatten()
dummies_actual_test = scaler.inverse_transform(dummies_actual_test)

new_y_test = dummies_actual_test[:, 0]

# Plot for test set
plt.figure(figsize=(10, 6))
plt.plot(new_y_test, label='Actual Close', color='blue')
plt.plot(test_predictions_unscaled, label='Predicted Close', color='orange')
plt.xlabel('Day')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Price (Test Set)')
plt.legend()
plt.savefig('actual_vs_predicted_close_test.png')

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


def create_sequences(data):
    xs, ys = [], []
    for i in range(len(data)):
        x = data[i,1:]  # Features
        y = data[i , 0]  # Target (Adj Close)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


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
# Example usage with your indices 


# Main Loop for Processing Stocks
errors = []
lookback = 180  # تنظیم مقدار lookback

# ایجاد فولدر برای ذخیره نتایج
lookback_dir = f'lookback_{lookback}'
plot_dir = os.path.join(lookback_dir, 'stock_plots') 
os.makedirs(plot_dir, exist_ok=True)

# فایل ارورها
error_file = os.path.join(lookback_dir, f'stock_errors_lookback_{lookback}.csv')

# خواندن فایل ارورها در صورت وجود
if os.path.exists(error_file):
    error_df = pd.read_csv(error_file)
    processed_stocks = error_df['Stock'].tolist()
else:
    processed_stocks = []

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
loss_function = torch.nn.MSELoss()

for stock_name in dataframes.keys():
    print(f"Processing stock: {stock_name}")
    if stock_name in processed_stocks:
        print(f"Skipping {stock_name}: Already processed.")
        continue  # رد سهامی که قبلاً پردازش شده‌اند
    
    try:
        # بررسی اینکه سهام داده‌ای در سال ۲۰۱۵ دارد یا خیر
        df = dc(dataframes[stock_name])
        df['Date'] = pd.to_datetime(df['Date'])
        df_2015 = df[(df['Date'] >= '2015-01-01') & (df['Date'] < '2016-01-01')]

        if len(df_2015) == 0:
            print(f"Skipping {stock_name}: No data available in 2015.")
            continue  # رد سهام‌هایی که در ۲۰۱۵ داده‌ای ندارند

        shifted_df = create_time_based_sliding_window(dataframes, lookback, stock_name, '1D', 3, -2, start_date='2016-01-01')

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

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

        # پیش‌بینی و محاسبه متریک‌ها برای تست
        with torch.no_grad():
            test_predictions = model(X_test.to(device)).cpu().numpy().flatten()

        mse = mean_squared_error(y_test.cpu().numpy().flatten(), test_predictions)
        mae = mean_absolute_error(y_test.cpu().numpy().flatten(), test_predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test.cpu().numpy().flatten(), test_predictions)

        # ذخیره ارورها
        errors.append([stock_name, mse, mae, rmse, mape])
        new_errors_df = pd.DataFrame(errors, columns=['Stock', 'MSE', 'MAE', 'RMSE', 'MAPE'])
        
        if os.path.exists(error_file):
            error_df = pd.read_csv(error_file)
            
            # ترکیب داده‌ها
            combined_df = pd.concat([error_df, new_errors_df], ignore_index=True)
            
            # حذف رکوردهای تکراری بر اساس ستون 'Stock'
            combined_df = combined_df.drop_duplicates(subset=['Stock'], keep='last')
        else:
            combined_df = new_errors_df
        # ذخیره داده‌ها در فایل CSV
        combined_df.to_csv(error_file, index=False)
        # بازگردانی پیش‌بینی‌ها به مقیاس اصلی
        dummies_test = np.zeros((X_test.shape[0], input_size + 1))
        dummies_test[:, 0] = test_predictions
        dummies_test = scaler.inverse_transform(dummies_test)
        test_predictions_unscaled = dummies_test[:, 0]

        dummies_actual_test = np.zeros((X_test.shape[0], input_size + 1))
        dummies_actual_test[:, 0] = y_test.cpu().numpy().flatten()
        dummies_actual_test = scaler.inverse_transform(dummies_actual_test)
        new_y_test = dummies_actual_test[:, 0]

        # ذخیره نمودارهای پیش‌بینی و واقعی برای تست
        plt.figure(figsize=(10, 6))
        plt.plot(new_y_test, label='Actual Close', color='blue')
        plt.plot(test_predictions_unscaled, label='Predicted Close', color='orange')
        plt.xlabel('Day')
        plt.ylabel('Close Price')
        plt.title(f'Actual vs Predicted Close Price (Test Set) for {stock_name}')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{stock_name}_lookback_{lookback}_actual_vs_predicted_close_test.png'))
        plt.close()

        # پیش‌بینی و ذخیره نمودارهای train
        with torch.no_grad():
            train_predictions = model(X_train.to(device)).cpu().numpy().flatten()

        dummies_train = np.zeros((X_train.shape[0], input_size + 1))
        dummies_train[:, 0] = train_predictions
        dummies_train = scaler.inverse_transform(dummies_train)
        train_predictions_unscaled = dummies_train[:, 0]

        dummies_actual_train = np.zeros((X_train.shape[0], input_size + 1))
        dummies_actual_train[:, 0] = y_train.cpu().numpy().flatten()
        dummies_actual_train = scaler.inverse_transform(dummies_actual_train)
        new_y_train = dummies_actual_train[:, 0]

        plt.figure(figsize=(10, 6))
        plt.plot(new_y_train, label='Actual Close', color='blue')
        plt.plot(train_predictions_unscaled, label='Predicted Close', color='orange')
        plt.xlabel('Day')
        plt.ylabel('Close Price')
        plt.title(f'Actual vs Predicted Close Price (Train Set) for {stock_name}')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{stock_name}_lookback_{lookback}_actual_vs_predicted_close_train.png'))
        plt.close()

        # پیش‌بینی و ذخیره نمودارهای validation
        with torch.no_grad():
            val_predictions = model(X_val.to(device)).cpu().numpy().flatten()

        dummies_val = np.zeros((X_val.shape[0], input_size + 1))
        dummies_val[:, 0] = val_predictions
        dummies_val = scaler.inverse_transform(dummies_val)
        val_predictions_unscaled = dummies_val[:, 0]

        dummies_actual_val = np.zeros((X_val.shape[0], input_size + 1))
        dummies_actual_val[:, 0] = y_val.cpu().numpy().flatten()
        dummies_actual_val = scaler.inverse_transform(dummies_actual_val)
        new_y_val = dummies_actual_val[:, 0]

        plt.figure(figsize=(10, 6))
        plt.plot(new_y_val, label='Actual Close', color='blue')
        plt.plot(val_predictions_unscaled, label='Predicted Close', color='orange')
        plt.xlabel('Day')
        plt.ylabel('Close Price')
        plt.title(f'Actual vs Predicted Close Price (Validation Set) for {stock_name}')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{stock_name}_lookback_{lookback}_actual_vs_predicted_close_validation.png'))
        plt.close()

        print(f"Completed processing for {stock_name}")

    except Exception as e:
        print(f"Error processing stock {stock_name}: {str(e)}")

print("All stocks processed.")
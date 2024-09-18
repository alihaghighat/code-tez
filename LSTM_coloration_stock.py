import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
directory_path = '100stocks'
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time

dataframes = {}
# List to store summary information
summary_data = []

for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        stock_name = filename.split('.')[0]
        file_path = os.path.join(directory_path, filename)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        dataframes[stock_name] = df
        
        # Get summary information
        num_records = len(df)
        start_date = df['Date'].min()
        end_date = df['Date'].max()
        
        # Append summary information to the list
        summary_info = {
            'Stock Name': stock_name,
            'Number of Records': num_records,
            'Start Date': start_date,
            'End Date': end_date
        }
        
        # Remove the 'Stock Name' key immediately
        del summary_info['Stock Name']
        
        # Append modified summary_info to summary_data
        summary_data.append(summary_info)

# Convert summary_data to a dictionary with stock name as key
summary_dict = {filename.split('.')[0]: item for item in summary_data}
#  Normalize each stock's dataframe
normalized_data = {}
for stock, df in dataframes.items():
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Min-max normalization
    df['Adj Close'] = (df['Adj Close'] - df['Adj Close'].min()) / (df['Adj Close'].max() - df['Adj Close'].min())
    
    # Save normalized data
    normalized_data[stock] = df[['Date', 'Adj Close']].rename(columns={'Adj Close': stock})

# Merge all normalized dataframes into a single DataFrame on 'Date'
merged_df = pd.concat([df.set_index('Date') for df in normalized_data.values()], axis=1, join='outer')

def split_stock_data(df, stock_name, n_delay=0):
    """
    Split the data for a specific stock into training, validation, and test sets, 
    starting from the first date where the stock has non-NaN values, with an optional delay.
    
    The entire DataFrame is returned (including all stocks) to ensure compatibility 
    with `sliding_window_with_top_correlations`, but the target stock is shifted by the given delay.
    
    :param df: DataFrame containing stock data (with Date as index)
    :param stock_name: Name of the target stock to process
    :param n_delay: Number of days to delay the start of the target stock data (default is 0)
    :return: Three DataFrames (train, validation, test) containing all stock data
    """
    # Ensure the stock exists in the DataFrame
    if stock_name not in df.columns:
        raise ValueError(f"Stock {stock_name} not found in DataFrame.")
    
    # Filter out rows where the target stock data is NaN
    df_filtered = df[df[stock_name].notna()]
    
    # Calculate the sizes of the training, validation, and test sets
    train_size = int(len(df_filtered) * 0.7)  # 70% for training
    val_size = int(len(df_filtered) * 0.15)   # 15% for validation
    
    # Split the data (all columns are retained, not just the target stock)
    train_data = df_filtered.iloc[:train_size]
    val_data = df_filtered.iloc[train_size:train_size + val_size]
    test_data = df_filtered.iloc[train_size + val_size:]
    
    return train_data, val_data, test_data

def pearson_correlation(x, y):
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    deviation_product = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = (sum((xi - mean_x) ** 2 for xi in x) / len(x)) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y) / len(y)) ** 0.5
    correlation = deviation_product / (len(x) * std_x * std_y)
    return correlation


def sliding_window_with_top_correlations(df, window_size, target_size, target_stock, n, d, step_size=1):
    """
    Sliding window with additional features: top N correlated stocks (with lag d) for the target stock.
    
    :param df: DataFrame containing stock data (with date as index, and each column representing a stock)
    :param window_size: Number of rows (days) in each window (input size)
    :param target_size: Number of rows (days) for the target (output size)
    :param target_stock: The column (stock) to compute the correlation against
    :param n: Number of top correlated stocks to use as features
    :param d: Number of days of lag to consider when calculating correlation
    :param step_size: Step size to move the window (default is 1)
    :return: Tuple of Tensors (inputs, targets), where:
             - inputs: Tensor of shape (num_windows, window_size, num_features)
             - targets: Tensor of shape (num_windows, target_size)
    """
    windows = []
    targets = []
    
    for i in range(0, len(df) - window_size - target_size - d + 1, step_size):
        # Get window data for target stock (sliding window) as input
        target_series = pd.to_numeric(df.iloc[i:i + window_size][target_stock], errors='coerce').values
        
        # Get the next 'target_size' data points after the window as target
        target_data = df.iloc[i + window_size:i + window_size + target_size][target_stock].values
        
        correlations = []
        for stock in df.columns:
            if stock != target_stock:
                # Ensure that stock data is numeric and lagged by 'd' days
                stock_series = pd.to_numeric(df.iloc[i:i + window_size - d][stock], errors='coerce').values  # Lagged by d days
                
                # Check for NaN values in the stock series and target series
                if np.isnan(stock_series).any() or np.isnan(target_series).any():
                    continue  # Skip this stock if any NaN values exist
                
                # Calculate correlation
                corr = pearson_correlation(stock_series, target_series)
                correlations.append((stock, corr))
        
        # Sort stocks by correlation and select top n correlated stocks
        top_n_correlated_stocks = sorted(correlations, key=lambda x: x[1], reverse=True)[:n]
        top_stock_names = [stock for stock, _ in top_n_correlated_stocks]
        
        # Collect data for the target stock and top n correlated stocks
        top_n_prices = [df.iloc[i:i + window_size][stock].values for stock in top_stock_names]
        
        # Create the combined window data: target stock + top n correlated stocks
        combined_window_data = np.column_stack((target_series, *top_n_prices))
        
        # Append the window data and the target data
        windows.append(combined_window_data)
        targets.append(target_data)
    
    # Convert the lists to numpy arrays, then to PyTorch tensors
    windows_np = np.array(windows, dtype=np.float32)  # Ensure windows are float32
    targets_np = np.array(targets, dtype=np.float32)  # Ensure targets are float32

    windows_tensor = torch.tensor(windows_np)
    targets_tensor = torch.tensor(targets_np)
    
    return windows_tensor, targets_tensor


# Measure the time for split_stock_data
print("Start split_stock_data ")
start_time = time.time()
train, val, test = split_stock_data(merged_df, stock_name='AAPL', n_delay=3)
end_time = time.time()
execution_time = end_time - start_time
print(f"End split_stock_data, Execution time: {execution_time:.2f} seconds")

# Measure the time for sliding window with top correlations for train
print("Start sliding window with top correlations for train")
start_time = time.time()
train_inputs, train_targets = sliding_window_with_top_correlations(
    df=train, 
    window_size=180, 
    target_size=1, 
    target_stock='AAPL', 
    n=3, 
    d=2, 
    step_size=1
)
end_time = time.time()
execution_time = end_time - start_time
print(f"End sliding window for train, Execution time: {execution_time:.2f} seconds")

# Measure the time for sliding window with top correlations for validation
print("Start sliding window with top correlations for validation")
start_time = time.time()
val_inputs, val_targets = sliding_window_with_top_correlations(
    df=val, 
    window_size=180, 
    target_size=1, 
    target_stock='AAPL', 
    n=3, 
    d=2, 
    step_size=1
)
end_time = time.time()
execution_time = end_time - start_time
print(f"End sliding window for validation, Execution time: {execution_time:.2f} seconds")

# Measure the time for sliding window with top correlations for test
print("Start sliding window with top correlations for test")
start_time = time.time()
test_inputs, test_targets = sliding_window_with_top_correlations(
    df=test, 
    window_size=180, 
    target_size=1, 
    target_stock='AAPL', 
    n=3, 
    d=2, 
    step_size=1
)
end_time = time.time()
execution_time = end_time - start_time
print(f"End sliding window for test, Execution time: {execution_time:.2f} seconds")

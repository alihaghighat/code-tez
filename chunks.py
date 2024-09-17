import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

directory_path = '100stocks'

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

def split_dataframe_with_delay_for_stocks(df, stockA_name, stockB_name, n, d, step):
    """
    Split the dataframe for two specific stocks into sliding window subsets with delay and return the result.

    :param df: DataFrame (with 'Date' as first column and stock data as other columns)
    :param stockA_name: Name of the first stock
    :param stockB_name: Name of the second stock
    :param n: Number of rows for each window
    :param d: Delay between the two windows
    :param step: Number of rows to move forward in each iteration
    :return: List of merged DataFrame containing stockA and stockB data
    """
    if stockA_name not in df.columns or stockB_name not in df.columns:
        raise ValueError(f"Stocks {stockA_name} or {stockB_name} not found in DataFrame.")
    
    # Ensure 'Date' is a column, not an index
    if 'Date' not in df.columns:
        df = df.reset_index()  # Convert index to column if necessary
    
    df_sorted = df.sort_values(by='Date')
    result = []

    # Loop through the DataFrame with the specified step size
    for i in range(0, len(df_sorted) - n - d + 1, step):
        stockA = df_sorted.loc[i:i + n - 1, ['Date', stockA_name]]  # Get stockA data for n rows
        stockB = df_sorted.loc[i + d:i + d + n - 1, ['Date', stockB_name]]  # Get stockB data with delay for n rows
        
        if len(stockA) == n and len(stockB) == n:
            # Merge stockA and stockB on 'Date'
            merged = pd.merge(stockA, stockB, on='Date', suffixes=('_stockA', '_stockB'))
            result.append(merged)
    
    return result

def split_dataframe_for_all_stock_pairs(df, n, d,step):
    """
    Apply split_dataframe_with_delay_for_stocks for each pair of stocks in the DataFrame.

    :param df: DataFrame (with 'Date' as first column and stock data as other columns)
    :param n: Number of rows for each split
    :param d: Delay between the two subsets
    :return: Dictionary with stock pair keys and list of merged DataFrame as values
    """
    # Get all combinations of two stocks
    stock_columns = df.columns.difference(['Date'])
    stock_pairs = list(combinations(stock_columns, 2))
    
    result = {}
  
    # Loop through each pair of stocks and apply split_dataframe_with_delay_for_stocks
    for stockA_name, stockB_name in stock_pairs:
        result[(stockA_name, stockB_name)] = split_dataframe_with_delay_for_stocks(df, stockA_name, stockB_name, n, d,step)
        
    return result

def sliding_window(df, window_size, step_size=1):
    """
    Apply sliding window technique to the given DataFrame.
    
    :param df: DataFrame containing stock data (with date as index)
    :param window_size: Number of rows (days) in each window
    :param step_size: Step size to move the window (default is 1)
    :return: List of tuples (window_data, target_data), where window_data is the 
             data in the window and target_data is the data following the window
    """
    windows = []
    
    # Loop over the DataFrame with the given step size
    for i in range(0, len(df) - window_size, step_size):
        # Get window data (sliding window)
        window_data = df.iloc[i:i + window_size].values
        
        # Get the next data point after the window (as target)
        target_data = df.iloc[i + window_size].values
        
        windows.append((window_data, target_data))
    
    return windows

train_windows = sliding_window(train, window_size, step_size)
val_windows = sliding_window(val, window_size, step_size)
test_windows = sliding_window(test, window_size, step_size)

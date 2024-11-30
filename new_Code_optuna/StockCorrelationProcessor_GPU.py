import pandas as pd
import numpy as np
import cupy as cp
from copy import deepcopy as dc
from multiprocessing import Pool, set_start_method

# Set multiprocessing to 'spawn' to avoid issues with CuPy
set_start_method('spawn', force=True)


# Define Pearson correlation using CuPy
def calculate_pearson_gpu(x, y):
    """
    Calculate Pearson correlation using CuPy on GPU.
    """
    x = cp.asarray(x)
    y = cp.asarray(y)
    if cp.std(x) == 0 or cp.std(y) == 0:
        return -1  # Return -1 if standard deviation is zero
    return cp.corrcoef(x, y)[0, 1].item()


# Helper function for multiprocessing
def process_row_multiprocessing(args):
    """
    Process a single row of data with GPU computations.
    """
    import cupy as cp  # Ensure CuPy is imported within the worker process
    row_index, stock_name, other_stocks_data, dataFrame, n_steps, delay = args
    correlations = []
    date_str = row_index.strftime('%Y-%m-%d')
    stock_data_1 = dataFrame[stock_name]

    for other_stock, other_stock_data in other_stocks_data.items():
        stock_data_2 = other_stock_data
        # Filter data for both stocks
        stock_data_1_filtered = stock_data_1[stock_data_1['Date'] <= date_str].reset_index(drop=True)
        stock_data_2_filtered = stock_data_2[stock_data_2['Date'] <= date_str].reset_index(drop=True)

        if len(stock_data_1_filtered) == 0 or len(stock_data_2_filtered) == 0:
            continue

        min_length = min(len(stock_data_1_filtered), len(stock_data_2_filtered))
        stock_data_1_filtered = stock_data_1_filtered.iloc[:min_length]
        stock_data_2_filtered = stock_data_2_filtered.iloc[:min_length]

        # Apply delays if needed
        if delay > 0:
            if len(stock_data_1_filtered) < n_steps + delay or len(stock_data_2_filtered) < n_steps:
                continue
            window_1 = stock_data_1_filtered['Adj Close'].iloc[-n_steps - delay:-delay].to_numpy()
            window_2 = stock_data_2_filtered['Adj Close'].iloc[-n_steps:].to_numpy()
        elif delay < 0:
            if len(stock_data_2_filtered) < n_steps + abs(delay) or len(stock_data_1_filtered) < n_steps:
                continue
            window_1 = stock_data_1_filtered['Adj Close'].iloc[-n_steps:].to_numpy()
            window_2 = stock_data_2_filtered['Adj Close'].iloc[-n_steps - abs(delay):-abs(delay)].to_numpy()
        else:
            if len(stock_data_1_filtered) < n_steps or len(stock_data_2_filtered) < n_steps:
                continue
            window_1 = stock_data_1_filtered['Adj Close'].iloc[-n_steps:].to_numpy()
            window_2 = stock_data_2_filtered['Adj Close'].iloc[-n_steps:].to_numpy()

        # Compute correlation using CuPy
        correlation = calculate_pearson_gpu(window_1, window_2)

        if correlation > 0:
            correlations.append((other_stock, correlation, stock_data_2_filtered[['Date', 'Adj Close']].iloc[-n_steps:]))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    top_correlated_stocks = correlations[:3]

    # Create new columns for the top correlated stocks
    new_columns = {f'Top_{i+1} Close(t-{j+1})': value
                   for i, (_, _, window_data) in enumerate(top_correlated_stocks)
                   for j, value in enumerate(window_data['Adj Close'].to_numpy()[::-1])}

    return row_index, new_columns


class StockCorrelationProcessor:
    def __init__(self, dataFrame):
        """
        Initialize the processor with the main stock DataFrame.
        """
        self.dataFrame = dc(dataFrame)

    def create_time_based_sliding_window(self, n_steps, stock_name, date_step='1D', n_top=3, delay=1, start_date='2018-01-01'):
        """
        Create a time-based sliding window with multiprocessing and GPU computation.
        """
        df = dc(self.dataFrame[stock_name])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Adj Close']]
        df.set_index('Date', inplace=True)
        df = df.resample(date_step).last().dropna()

        # Add shifted columns for sliding window
        shifted_columns = [df['Adj Close'].shift(365 - i).rename(f'Adj Close(t-{i})') for i in range(1, n_steps + 1)]
        df = pd.concat([df] + shifted_columns, axis=1).dropna()

        # Prepare data for other stocks
        other_stocks_data = {col: self.dataFrame[col].copy() for col in self.dataFrame.keys() if col != stock_name}
        for stock, stock_df in other_stocks_data.items():
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])
            other_stocks_data[stock] = stock_df

        start_date = pd.to_datetime(start_date)
        df = df[df.index >= start_date]

        # Prepare arguments for multiprocessing
        args = [(row_index, stock_name, other_stocks_data, self.dataFrame, n_steps, delay) for row_index in df.index]

        # Use multiprocessing Pool
        all_new_columns = {}
        with Pool() as pool:
            results = pool.map(process_row_multiprocessing, args)

        # Combine results
        for row_index, new_columns in results:
            if new_columns:
                all_new_columns[row_index] = new_columns

        new_columns_df = pd.DataFrame.from_dict(all_new_columns, orient='index')
        df = pd.concat([df, new_columns_df], axis=1)

        return df

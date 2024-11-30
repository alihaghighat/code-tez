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
import concurrent.futures
def save_test_results_to_csv(stock_name, lookback, delay, n_top, mse, mae, rmse, mape, filename="test_results.csv"):
    data = {
        "Stock": [stock_name],
        "Lookback": [lookback],
        "Delay": [delay],
        "N_Top": [n_top],
        "MSE": [mse],
        "MAE": [mae],
        "RMSE": [rmse],
        "MAPE": [mape]
    }
    df = pd.DataFrame(data)
    with open(filename, mode='a') as f:
        df.to_csv(f, header=f.tell() == 0, index=False)


def split_data(X, y, train_ratio=0.7, validation_ratio=0.15):
    # تقسیم داده‌ها به آموزش و بقیه (اعتبارسنجی و تست)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_ratio, shuffle=True)

    # تقسیم داده‌های باقی‌مانده به اعتبارسنجی و تست
    val_ratio_adjusted = validation_ratio / (1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_ratio_adjusted, shuffle=True)

    return X_train, y_train, X_val, y_val, X_test, y_test
# بارگذاری مدل با بهترین پارامترها و آماده‌سازی داده‌های تست
# بارگذاری فایل min_error_per_stock.csv برای مقداردهی اولیه delay
min_error_df = pd.read_csv("./min_error_per_stock.csv")  # مطمئن شوید که مسیر فایل درست است

# تابع برای دریافت مقدار delay از فایل بر اساس نام سهام
def get_initial_delay(stock_name, min_error_df):
    row = min_error_df[min_error_df["Stock"] == stock_name]
    if not row.empty:
        return int(row["Best Delay"].values[0])  # تبدیل مقدار به عدد صحیح
    return -3  # مقدار پیش‌فرض در صورت نبودن سهام در فایل


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
# خواندن فایل stocks_server_1.csv برای لیست سهام‌هایی که باید تحلیل شوند
stocks_server_1_df = pd.read_csv('../stocks_server_1.csv')
stocks_server_1 = stocks_server_1_df['Stock'].tolist()
print(stocks_server_1)
# مسیر پوشه‌ای که فایل‌های CSV سهام‌ها در آن قرار دارند
directory_path = '../100stocks'
dataframes = {}

# به صورت هم‌زمان فایل‌های CSV را پردازش کنید
def load_stock_data(filename, stocks_server_1, directory_path):
    stock_name = filename.split('.')[0]
    if stock_name in stocks_server_1:
        file_path = os.path.join(directory_path, filename)
        return stock_name, pd.read_csv(file_path)
    return None

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(lambda filename: load_stock_data(filename, stocks_server_1, directory_path), os.listdir(directory_path))
    dataframes = {result[0]: result[1] for result in results if result}
print(dataframes)

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
    shifted_columns = [df['Adj Close'].shift(365 - i).rename(f'Adj Close(t-{i})') for i in  range(1, n_steps + 1)]
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


def train_one_epoch(epoch, model, train_loader, optimizer, loss_function, device):
    model.train()
    running_loss = 0.0
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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

# تنظیمات اولیه
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
hidden_size = 200
num_layers = 3
dropout_rate = 0.2
learning_rate = 0.0001
num_epochs = 100
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15
loss_function = torch.nn.MSELoss()
def evaluate_best_model(dataframes, best_params, best_model_state, stock_name='MSFT', device='cpu', batch_size=32):
    lookback, delay, n_top = best_params
    
    # پیش‌پردازش داده‌ها با بهترین پارامترها
    shifted_df = create_time_based_sliding_window(dataframes, lookback, stock_name, '1D', n_top, delay, start_date='2016-01-01')
    shifted_df_as_np = shifted_df.to_numpy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(shifted_df_as_np)

    X, y = create_sequences(scaled_data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, train_ratio=0.7, validation_ratio=0.15)
    X_test, y_test = reshape_data(X_test, y_test, lookback)
    test_dataset = TimeSeriesDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # بارگذاری مدل ذخیره‌شده و انجام تست
    model = LSTM(X_test.shape[2], hidden_size, num_layers, dropout_rate).to(device)
    model.load_state_dict(best_model_state)

    # اجرای تست و محاسبه نتایج
    mse, mae, rmse, mape = test_model(model, test_loader, device)

    # ذخیره نتایج تست و پارامترها در CSV
    save_test_results_to_csv(stock_name, lookback, delay, n_top, mse, mae, rmse, mape)

    print(f"Test results saved: Stock={stock_name}, Lookback={lookback}, Delay={delay}, N_Top={n_top}, MSE={mse}, MAE={mae}, RMSE={rmse}, MAPE={mape}")


# تعریف مدل LSTM
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


# تعریف مدل بهینه‌سازی پارامترها (شبکه عصبی کمک‌کننده)
class ParameterOptimizer(nn.Module):
    def __init__(self):
        super(ParameterOptimizer, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # خروجی 3 برای lookback، delay و n_top

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = torch.sigmoid(self.fc3(x))  # محدوده خروجی را به [0, 1] محدود می‌کنیم
        return out  

# تولید پارامترهای اولیه با استفاده از ParameterOptimizer
def generate_initial_parameters(param_optimizer):
    random_input = torch.rand(1, 10)  # ورودی تصادفی
    output = param_optimizer(random_input)
    
    # تبدیل خروجی به مقادیر عددی مورد نظر
    lookback = int(output[0][0] * 20 + 20)  # محدوده: 20 تا 40
    delay = int(output[0][1] * -4 - 1)  # محدوده: -1 تا -5
    n_top = int(output[0][2] * 2 + 2)  # محدوده: 2 تا 4
    return lookback, delay, n_top

# تابعی برای تغییر جزئی پارامترهای بهترین ترکیب قبلی
def mutate_parameters(best_params, scale=0.3):
    lookback, delay, n_top = best_params
    lookback = max(20, min(40, int(lookback + np.random.normal(0, scale * lookback))))
    delay = max(-5, min(-1, int(delay + np.random.normal(0, scale * abs(delay)))))
    n_top = max(2, min(4, int(n_top + np.random.normal(0, scale * n_top))))
    return lookback, delay, n_top
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


# تعریف Dataset برای سری زمانی
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# حلقه آموزش و ارزیابی برای همه سهام‌ها
def train_model_for_all_stocks(dataframes, param_optimizer, num_epochs=100, patience=30, top_n=5):
    for stock_name, stock_data in dataframes.items():
        print(f"Processing stock: {stock_name}")
        
        best_loss = float('inf')
        best_model_state = None
        initial_delay = get_initial_delay(stock_name, min_error_df)
        print(initial_delay)
        best_params = (30, initial_delay, 3)  # مقداردهی اولیه با توجه به مقدار delay از فایل
        no_improve_count = 0
        cache = {}

        for epoch in range(num_epochs):
            # استفاده از mutate_parameters برای تولید پارامترهای جدید
            lookback, delay, n_top = mutate_parameters(best_params)
            print(f"Epoch {epoch + 1}: Testing with lookback={lookback}, delay={delay}, n_top={n_top}")

            cache_key = (lookback, delay, n_top)
            if cache_key in cache:
                shifted_df = cache[cache_key]
                print(f"Using cached data for parameters: {cache_key}")
            else:
                shifted_df = create_time_based_sliding_window(dataframes, lookback, stock_name, '1D', n_top, delay, start_date='2016-01-01')
                if shifted_df.empty:
                    continue
                cache[cache_key] = shifted_df

            shifted_df_as_np = shifted_df.to_numpy()
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(shifted_df_as_np)

            X, y = create_sequences(scaled_data)
            X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, train_ratio=0.7, validation_ratio=0.15)
        # After reshaping data
            X_train, y_train = reshape_data(X_train, y_train, lookback)
            X_val, y_val = reshape_data(X_val, y_val, lookback)
            X_test, y_test = reshape_data(X_test, y_test, lookback)

            # Add these assertions to check for NaN values in the data
            assert not np.isnan(X_train).any(), "NaN values found in training data"
            assert not np.isnan(X_val).any(), "NaN values found in validation data"
            assert not np.isnan(X_test).any(), "NaN values found in test data"
            assert not np.isnan(y_train).any(), "NaN values found in training labels"
            assert not np.isnan(y_val).any(), "NaN values found in validation labels"
            assert not np.isnan(y_test).any(), "NaN values found in test labels"

            # Continue with creating DataLoader objects
            train_loader = DataLoader(TimeSeriesDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float()), batch_size=32, shuffle=True)
            val_loader = DataLoader(TimeSeriesDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float()), batch_size=32, shuffle=False)
            test_loader = DataLoader(TimeSeriesDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float()), batch_size=32, shuffle=False)

            model = LSTM(X_train.shape[2], 200, 3, 0.2).to(device)
            # Initialize the weights of the model
            model.apply(initialize_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Try a smaller learning rate
            loss_function = torch.nn.MSELoss()

            for i in range(5):
                train_one_epoch(i, model, train_loader, optimizer, loss_function, device)
                val_loss = validate_one_epoch(model, val_loader, loss_function, device)

            # ذخیره نتایج ارزیابی و پارامترها در فایل CSV
            mse, mae, rmse, mape = test_model(model, test_loader, device)
            save_results_to_csv(stock_name, epoch + 1, lookback, delay, n_top, val_loss, mse, mae, rmse, mape)

            # به‌روزرسانی بهترین مدل
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict()
                best_params = (lookback, delay, n_top)
                no_improve_count = 0
                print(f"Updated best model for {stock_name}: Lookback={lookback}, Delay={delay}, N_Top={n_top}, Validation Loss={val_loss}")
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(f"Early stopping for {stock_name} due to no improvement")
                break

        # ارزیابی بهترین مدل برای هر سهام پس از آموزش
        if best_model_state is not None:
            evaluate_best_model(dataframes, best_params, best_model_state, stock_name, device=device, batch_size=32)

        print(f"Finished processing stock: {stock_name}")
# فراخوانی تابع برای پردازش تمامی سهام‌ها
def save_results_to_csv(stock_name, epoch, lookback, delay, n_top, val_loss, mse, mae, rmse, mape, filename="results.csv"):
    data = {
        "Stock": [stock_name],
        "Epoch": [epoch],
        "Lookback": [lookback],
        "Delay": [delay],
        "N_Top": [n_top],
        "Validation Loss": [val_loss],
        "MSE": [mse],
        "MAE": [mae],
        "RMSE": [rmse],
        "MAPE": [mape]
    }
    df = pd.DataFrame(data)
    with open(filename, mode='a') as f:
        df.to_csv(f, header=f.tell() == 0, index=False)




class EarlyStopping:
    def __init__(self, patience=30, min_delta=0):
        """
        :param patience: تعداد دوره‌هایی که بهبود نداشته باشد تا متوقف شود
        :param min_delta: کمترین تغییر مورد انتظار برای در نظر گرفتن بهبود
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# تابع برای تست مدل
def test_model(model, test_loader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            predictions.extend(output.cpu().numpy().flatten())
            actuals.extend(y_batch.cpu().numpy().flatten())
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actuals, predictions)
    print(f"Test Results - MSE: {mse}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}")
    return mse, mae, rmse, mape

# ایجاد و آموزش مدل بهینه‌ساز پارامترها
param_optimizer = ParameterOptimizer()
train_model_for_all_stocks(dataframes, param_optimizer, num_epochs=100, patience=30, top_n=5)

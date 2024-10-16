import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
directory_path = '100stocks'
from torch.utils.data import DataLoader, TensorDataset
import optuna

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


# نمایش لیست ستون‌ها

def split_stock_data(df, stock_name, n_delay=0):
    """
    Split the data for a specific stock into training, validation, and test sets, 
    starting from the first date where the stock has non-NaN values, with an optional delay.
    
    :param df: DataFrame containing stock data
    :param stock_name: Name of the stock to be processed
    :param n_delay: Number of days to delay the start of the data (default is 0)
    :return: Three DataFrames (train, validation, test)
    """
    # Ensure the stock exists in the dataframe
    if stock_name not in df.columns:
        raise ValueError(f"Stock {stock_name} not found in DataFrame.")
    
    # Filter out rows where the stock data is NaN
    df_filtered = df[df[stock_name].notna()]
    
    # Ensure there are enough data points after delay
    if len(df_filtered) <= n_delay:
        raise ValueError(f"Not enough data for stock {stock_name} after applying delay.")
    
    # Apply the delay by skipping the first n_delay rows
    df_filtered = df_filtered.iloc[n_delay:]
    
    # Ensure there are still enough data points after applying delay
    if len(df_filtered) < 10:
        raise ValueError(f"Not enough data for stock {stock_name} after applying delay.")
    
    # Only keep the stock data column (index is already 'Date')
    df_filtered = df_filtered[[stock_name]]
    
    # Calculate the sizes of the training, validation, and test sets
    train_size = int(len(df_filtered) * 0.7)  # 70% for training
    val_size = int(len(df_filtered) * 0.15)   # 15% for validation
    
    # Split the data
    train_data = df_filtered.iloc[:train_size]
    val_data = df_filtered.iloc[train_size:train_size + val_size]
    test_data = df_filtered.iloc[train_size + val_size:]
    
    return train_data, val_data, test_data
   

def sliding_window(df, window_size, target_size, step_size=1):
    """
    Apply sliding window technique to the given DataFrame for multi-step target prediction.
    
    :param df: DataFrame containing stock data (with date as index)
    :param window_size: Number of rows (days) in each window (input size)
    :param target_size: Number of rows (days) for the target (output size)
    :param step_size: Step size to move the window (default is 1)
    :return: Tuple of Tensors (inputs, targets), where:
             - inputs: Tensor of shape (num_windows, window_size, num_features)
             - targets: Tensor of shape (num_windows, target_size, num_features)
    """
    windows = []
    targets = []
    
    # Loop over the DataFrame with the given step size
    for i in range(0, len(df) - window_size - target_size + 1, step_size):
        # Get window data (sliding window) as input
        window_data = df.iloc[i:i + window_size].values
        
        # Get the next 'target_size' data points after the window as target
        target_data = df.iloc[i + window_size:i + window_size + target_size].values
        
        windows.append(window_data)
        targets.append(target_data)
    
    # Convert the lists to numpy arrays, then to PyTorch tensors
    windows_tensor = torch.tensor(windows, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    return windows_tensor, targets_tensor


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, target_size, output_size, dropout=0.5, l2_lambda=0.01):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, target_size * output_size)
        self.target_size = target_size
        self.output_size = output_size
        self.l2_lambda = l2_lambda  # L2 regularization lambda value
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        out = out.view(-1, self.target_size, self.output_size)
        return out

    def l2_regularization_loss(self):
        l2_loss = 0
        for param in self.parameters():
            if param.requires_grad and param.ndimension() > 1:  # Ignore bias terms
                l2_loss += torch.norm(param, 2)
        return self.l2_lambda * l2_loss

# EarlyStopping Class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience  # Number of epochs to wait for improvement
        self.min_delta = min_delta  # Minimum change in loss to be considered as improvement
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def check(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset patience
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
# Function to train the model
def train_model(model, train_loader, val_loader, num_epochs, device, optimizer, criterion, early_stopping):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        # Training loop
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets) + model.l2_regularization_loss()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation loss calculation
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0

        # Check if val_loader is not empty before calculating validation loss
        if len(val_loader) > 0:
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            # Compute average loss for the epoch
            avg_val_loss = val_loss / len(val_loader)
        else:
            print("Warning: Validation loader is empty.")
            avg_val_loss = float('inf')  # Assign a large value if validation data is empty

        avg_train_loss = running_loss / len(train_loader)

        # Print average loss for training and validation
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.10f}, Validation Loss: {avg_val_loss:.10f}')

        # Check early stopping
        early_stopping.check(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Function to test the model
def test_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0

    # Check if test_loader is not empty
    if len(test_loader) > 0:
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        # Compute average test loss
        average_test_loss = test_loss / len(test_loader)
    else:
        print("Warning: Test loader is empty.")
        average_test_loss = float('inf')  
    return average_test_loss



delay_day_target_stock=2
train, val, test = split_stock_data(merged_df, 'AAPL',delay_day_target_stock)  
window_size = 180  
target_size = 5  
step_size = 3      

# Apply sliding window on training, validation, and test data
train_inputs, train_targets = sliding_window(train, window_size, target_size, step_size)
val_inputs, val_targets = sliding_window(val, window_size, target_size, step_size)
test_inputs, test_targets = sliding_window(test, window_size, target_size, step_size)

# print(f"Train inputs shape: {train_inputs.shape}")   # Should be (num_windows, window_size, num_features)
# print(f"Train targets shape: {train_targets.shape}") # Should be (num_windows, target_size, num_features)

train_dataset = TensorDataset(train_inputs, train_targets)
val_dataset = TensorDataset(val_inputs, val_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

# Create DataLoader for batching
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Hyperparameters

input_size = train_inputs.shape[2]  # Number of features (e.g., number of stocks)
hidden_size = 256  # Increased hidden size for potentially better learning
num_layers = 3  # Increase LSTM layers to make the model deeper
output_size = train_inputs.shape[2]  # Number of features to predict (e.g., predicting for multiple stocks)
target_size = train_targets.shape[1]  # Number of days to predict (e.g., 5 days ahead)
dropout = 0.5  # Dropout to reduce overfitting
l2_lambda = 0.001  # L2 Regularization
learning_rate = 0.0005  # Lower learning rate for better convergence
num_epochs = 100
patience = 10  # Patience for early stopping

# Initialize the model with regularization
model = StockLSTM(input_size, hidden_size, num_layers, target_size, output_size, dropout=dropout, l2_lambda=l2_lambda)
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Early stopping initialization
early_stopping = EarlyStopping(patience=patience)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# تعریف متغیری برای ذخیره بهترین Loss و مدل
best_model = None
best_val_loss = float('inf')  # بهترین loss را به یک عدد بزرگ مقداردهی اولیه کنید
def objective(trial):
    global best_model, best_val_loss  # استفاده از متغیرهای سراسری برای ذخیره بهترین مدل
    
    # انتخاب سهام‌های مختلف
    stock_symbols = merged_df.columns[merged_df.notna().sum() >= 500] # لیستی از نمادهای سهام
    stock_symbol = trial.suggest_categorical('stock_symbol', stock_symbols)  # انتخاب سهام
    
    # تقسیم داده‌ها برای سهام انتخاب‌شده
    train_data, val_data, test_data = split_stock_data(merged_df, stock_symbol, delay_day_target_stock)
    
    # انتخاب هایپرپارامترهای مختلف
    dropout = trial.suggest_float('dropout', 0.3, 0.7)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-2)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
    num_epochs = trial.suggest_int('num_epochs', 10, 100)
    window_size = trial.suggest_int('window_size', 60, 180)  # انتخاب اندازه پنجره
    target_size = trial.suggest_int('target_size', 1, 5)  # انتخاب تعداد روزهای هدف
    
    # اعمال sliding window روی داده‌های آموزشی، اعتبارسنجی و تست
    step_size = 1  # می‌توانید step_size را به‌عنوان پارامتر تنظیم کنید
    train_inputs, train_targets = sliding_window(train_data, window_size, target_size, step_size)
    val_inputs, val_targets = sliding_window(val_data, window_size, target_size, step_size)
    test_inputs, test_targets = sliding_window(test_data, window_size, target_size, step_size)


    # ساخت TensorDataset و DataLoader
    batch_size = 32
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ساخت مدل با پارامترهای انتخاب‌شده
    model = StockLSTM(input_size, hidden_size, num_layers, target_size, output_size, dropout=dropout, l2_lambda=l2_lambda)

    # انتقال مدل به GPU (اگر موجود باشد)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # تعریف بهینه‌ساز با استفاده از learning_rate و l2_lambda (weight decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # تعریف EarlyStopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    # آموزش مدل با داده‌های sliding window
    train_model(model, train_loader, val_loader, num_epochs, device, optimizer, criterion, early_stopping)
    
    # اعتبارسنجی مدل
    val_loss = test_model(model, test_loader, device)

    # بررسی اینکه val_loss None نیست و سپس مقایسه انجام دهید
    if val_loss is not None and val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model  # ذخیره بهترین مدل

    return val_loss if val_loss is not None else float('inf')  # بازگشت loss برای بهینه‌سازی، اگر None باشد یک عدد بزرگ برمی‌گرداند

# اجرای بهینه‌سازی
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# ذخیره بهترین مدل بعد از بهینه‌سازی
torch.save(best_model.state_dict(), 'best_model_One_stock.pth')  # ذخیره وزن‌های بهترین مدل

print("Best trial:", study.best_trial)

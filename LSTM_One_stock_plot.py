import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
directory_path = '100stocks'
from torch.utils.data import DataLoader, TensorDataset
import optuna
import matplotlib.pyplot as plt

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
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Compute average loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Print average loss for training and validation
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.10f}, Validation Loss: {avg_val_loss:.10f}')

    

# Function to test the model
def test_model(model, test_loader, device):
    model.eval()  # Change to eval mode to prevent dropout and batchnorm from being applied
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()  # Add loss for each batch

    average_test_loss = test_loss / len(test_loader)  # Calculate average test loss
    print(f'Test Loss: {average_test_loss:.10f}')
    return average_test_loss  # Return average test loss



delay_day_target_stock=2
train, val, test = split_stock_data(merged_df, 'AAPL',delay_day_target_stock)  
print(val)
window_size = 180  
target_size = 1 
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
num_layers = 4  # Increase LSTM layers to make the model deeper
output_size = train_inputs.shape[2]  # Number of features to predict (e.g., predicting for multiple stocks)
target_size = train_targets.shape[1]  # Number of days to predict (e.g., 5 days ahead)
# Params from Optuna
dropout = 0.5 # Dropout from Optuna
l2_lambda = 2.7985336558575806e-05  # L2 Regularization from Optuna
learning_rate = 0.0001  # Learning rate from Optuna
num_epochs = 100  # Number of epochs from Optuna

# Initialize the model with the suggested parameters from Optuna
model = StockLSTM(input_size, hidden_size, num_layers, target_size, output_size, dropout=dropout, l2_lambda=l2_lambda)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Early stopping initialization
patience = 10  # Patience for early stopping remains the same
early_stopping = EarlyStopping(patience=patience)

# Define loss function and optimizer with the Optuna-suggested learning rate and L2 regularization
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)  # Adam with weight decay (L2 regularization)


def test_and_plot(model, test_loader, stock, device, criterion, dataframes, save_path='actual_vs_predicted.png'):
    model.eval()  # Set the model to evaluation mode
    actuals = []
    predictions = []
    test_loss = 0.0

    # Calculate the test loss and perform predictions in a single loop
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass to get model predictions
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()  # Add loss for each batch

            # Collect actual and predicted values
            actuals.append(targets.cpu().numpy())
            predictions.append(outputs.cpu().numpy())

    # Calculate average test loss
    average_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {average_test_loss:.10f}')

    # Convert lists to numpy arrays
    actuals = np.concatenate(actuals, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    # Ensure the stock exists in the dataframes and fetch min/max values for denormalization
    if stock in dataframes:
        df = dataframes[stock]
        min_value = df['Adj Close'].min()
        max_value = df['Adj Close'].max()
    else:
        raise ValueError(f"Stock {stock} not found in dataframes.")

    # Denormalize the actual and predicted values
    denormalized_actuals = denormalize(actuals, min_value, max_value)
    denormalized_predictions = denormalize(predictions, min_value, max_value)

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(denormalized_actuals[:, 0, 0], label='Actual Prices')
    plt.plot(denormalized_predictions[:, 0, 0], label='Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'Actual vs Predicted Stock Prices for {stock}')
    plt.legend()

    # Save the plot to a file
    plt.savefig(save_path)
    plt.show()

    # Return actual and predicted values for further use if needed
    return denormalized_actuals, denormalized_predictions

def denormalize(data, min_value, max_value):
    return data * (max_value - min_value) + min_value
# Train the model
train_model(model, train_loader, val_loader, num_epochs, device, optimizer, criterion, early_stopping)

# Test the model and plot the results
actuals, predictions = test_and_plot(model, test_loader, 'AAPL', device, criterion, dataframes, save_path='actual_vs_predicted_prices_LSTM_One_stock_plot.png')

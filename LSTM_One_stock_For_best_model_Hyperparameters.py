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

        # Check early stopping
        early_stopping.check(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

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


# Define a variable to store the best loss and model
best_model = None
best_val_loss = float('inf')  # Initialize best loss to a large value

def objective(trial):
    global best_model, best_val_loss  # Use global variables to store the best model
    
    # Suggest different hyperparameters
    dropout = trial.suggest_float('dropout', 0.3, 0.7)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-2)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)

    # Suggest different numbers of epochs
    num_epochs = trial.suggest_int('num_epochs', 80, 100)  # e.g., between 80 to 100

    # Build the model with the suggested parameters
    model = StockLSTM(input_size, hidden_size, num_layers, target_size, output_size, dropout=dropout, l2_lambda=l2_lambda)

    # Move the model to GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the optimizer using the suggested learning_rate and l2_lambda (weight decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # Define EarlyStopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    # Train the model with the suggested number of epochs
    train_model(model, train_loader, val_loader, num_epochs, device, optimizer, criterion, early_stopping)
    
    # Validate the model
    val_loss = test_model(model, val_loader, device)

    # Check if val_loss is not None and then compare
    if val_loss is not None and val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model  # Store the best model

    return val_loss if val_loss is not None else float('inf')  # Return loss for optimization, if None return a large number

# Run the optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Save the best model after optimization
torch.save(best_model.state_dict(), 'best_model_One_stock.pth')  # Save the best model's weights

print("Best trial:", study.best_trial)

# Load the best model
best_model = StockLSTM(input_size, hidden_size, num_layers, target_size, output_size, dropout=dropout, l2_lambda=l2_lambda)
best_model.load_state_dict(torch.load('best_model_One_stock.pth'))
best_model = best_model.to(device)  # Move the model to GPU if available

# Function to plot actual vs predicted prices
def plot_actual_vs_predicted(model, test_loader, device, save_path='actual_vs_predicted.png'):
    model.eval()  # Set the model to evaluation mode to disable dropout and batchnorm
    actuals = []
    predictions = []
    
    with torch.no_grad():  # Disable gradient calculation for testing
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            # Move data back to CPU for plotting
            actuals.append(targets.cpu().numpy())
            predictions.append(outputs.cpu().numpy())

    # Convert lists to numpy arrays
    actuals = np.concatenate(actuals, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    # Plot actual vs predicted values for the first target feature
    plt.figure(figsize=(10, 6))
    plt.plot(actuals[:, 0, 0], label='Actual Prices')
    plt.plot(predictions[:, 0, 0], label='Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.legend()

    # Save the plot to a file
    plt.savefig(save_path)
    plt.show()

    return actuals, predictions  # Return actual and predicted values for further use

# Plot and save the actual vs predicted prices
actuals, predictions = plot_actual_vs_predicted(best_model, test_loader, device, save_path='actual_vs_predicted_prices.png')

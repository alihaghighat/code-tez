import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
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


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on: {device}')

# Assuming you have a DataFrame with 'Date' and 'AAPL' columns
# Drop the 'Date' column and keep only the 'AAPL' values
normalized_data_AAPL= normalized_data['AAPL'].drop(columns=['Date'])
data = normalized_data_AAPL.values.astype(np.float32)
# Set a seed for reproducibility



# Split the data into training, validation, and test sets
train_size = int(len(data) * 0.7)  # 70% for training
val_size = int(len(data) * 0.15)   # 15% for validation
test_size = len(data) - train_size - val_size  # Remaining 15% for test

# Split the data
train_data = data[:train_size]
val_data = data[train_size:train_size+val_size]
test_data = data[train_size+val_size:]

# Function to create sequences for the LSTM model
def create_sequences(data, seq_length, n_days_ahead):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - n_days_ahead):
        x = data[i:(i+seq_length)]
        y = data[(i+seq_length):(i+seq_length+n_days_ahead)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 180  # Length of the sequence for the LSTM
n_days_ahead = 1  # Number of days ahead to predict

# Create sequences for training, validation, and test data
X_train, y_train = create_sequences(train_data, seq_length, n_days_ahead)
X_val, y_val = create_sequences(val_data, seq_length, n_days_ahead)
X_test, y_test = create_sequences(test_data, seq_length, n_days_ahead)

# Convert data to PyTorch tensors and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
X_val = torch.from_numpy(X_val).float().to(device)
y_val = torch.from_numpy(y_val).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, n_days_ahead=5):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, n_days_ahead)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Initialize the model, loss function, and optimizer
model = LSTM(input_size=1, hidden_layer_size=50, n_days_ahead=n_days_ahead).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop with validation
epochs = 100
for epoch in range(epochs):
    model.train()  # Set the model to training mode

    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                             torch.zeros(1, 1, model.hidden_layer_size).to(device))

        y_pred = model(seq.unsqueeze(0))  # Add batch dimension
        labels = labels.view(1, -1)  # Reshape labels to (1, n_days_ahead)
        single_loss = loss_function(y_pred, labels)

        single_loss.backward()
        optimizer.step()

    # Validation after each epoch
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_loss = 0
        for seq, labels in zip(X_val, y_val):
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                 torch.zeros(1, 1, model.hidden_layer_size).to(device))

            y_pred = model(seq.unsqueeze(0))
            labels = labels.view(1, -1)
            val_loss += loss_function(y_pred, labels).item()

        val_loss /= len(X_val)

    # Print the training and validation loss every 10 epochs
    if epoch % 10 == 1:
        print(f'Epoch {epoch:3} Training Loss: {single_loss.item():10.10f} Validation Loss: {val_loss:10.10f}')

# Test the model after training
model.eval()
with torch.no_grad():
    test_loss = 0
    for seq, labels in zip(X_test, y_test):
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                             torch.zeros(1, 1, model.hidden_layer_size).to(device))

        y_pred = model(seq.unsqueeze(0))
        labels = labels.view(1, -1)
        print(loss_function(y_pred, labels).item())
        test_loss += loss_function(y_pred, labels).item()
        

    test_loss /= len(X_test)
    print(f'Test Loss: {test_loss:10.8f}')

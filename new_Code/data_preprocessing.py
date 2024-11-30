import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
class DataPreprocessor:
    """
    A class for data preprocessing tasks, including splitting data,
    creating sequences, and reshaping data for time-series models.
    """

    def __init__(self):
        pass




    @staticmethod
    def split_data(X, y, train_ratio=0.7, validation_ratio=0.15):
        """
        Split the data into training, validation, and test sets.

        Parameters:
            X (ndarray): Features.
            y (ndarray): Labels/targets.
            train_ratio (float): Ratio of training data.
            validation_ratio (float): Ratio of validation data.

        Returns:
            tuple: Split data (X_train, y_train, X_val, y_val, X_test, y_test).
        """
        # Split data into training and remaining sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_ratio, shuffle=True)

        # Split remaining data into validation and test sets
        val_ratio_adjusted = validation_ratio / (1 - train_ratio)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_ratio_adjusted, shuffle=True)

        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def create_sequences(data):
        """
        Create sequences from data for time-series models.

        Parameters:
            data (ndarray): Input data.

        Returns:
            tuple: Sequences (xs) and corresponding labels (ys).
        """
        xs, ys = [], []
        for i in range(len(data)):
            x = data[i, 1:]  # Features
            y = data[i, 0]  # Target (Adj Close)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    
    def reshape_data(self, X, y, lookback):
        if X.shape[0] <= 0 or X.shape[1] <= 0:
            raise ValueError(f"Invalid X dimensions for reshaping: {X.shape}")

        if lookback <= 0:
            raise ValueError(f"Invalid lookback value: {lookback}")

        num_samples, num_features = X.shape[0], X.shape[1]
        X_reshaped = np.zeros((num_samples, lookback, num_features))
        
        # Reshaping logic here
        return X_reshaped, y

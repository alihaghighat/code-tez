import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import optuna
from functools import partial
from load_all_stocks import StockDataLoader
from StockCorrelationProcessor_GPU import StockCorrelationProcessor
from data_preprocessing import DataPreprocessor, TimeSeriesDataset
from model import LSTM
from training import train_one_epoch, validate_one_epoch, test_model, EarlyStopping
from results_saver import ResultsSaver


class StockTrainingPipelineWithOptuna:
    def __init__(self, directory_path='./100stocks', min_error_file="./min_error_per_stock.csv", num_epochs=150, num_epochs_for_stock=200, patience=50, 
                 batch_size=32, hidden_size=200, num_layers=5, dropout_rate=0.2, learning_rate=0.00001):
        self.directory_path = directory_path
        self.min_error_file = min_error_file
        self.num_epochs = num_epochs
        self.num_epochs_for_stock = num_epochs_for_stock
        self.patience = patience
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.set_seed(42)

        self.loader = StockDataLoader(self.directory_path)
        self.results_saver = ResultsSaver(test_results_file='test_results.csv')
        self.preprocessor = DataPreprocessor()

        self.loader.load_all_stocks()
        self.dataframes = self.loader.get_dataframes()

        # Load the min error file into a DataFrame
        self.min_error_df = pd.read_csv(self.min_error_file)
        self.best_params_file = "best_params_per_stock.csv"

    def set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_initial_delay(self, stock_name):
        row = self.min_error_df[self.min_error_df["Stock"] == stock_name]
        if not row.empty:
            return int(row["Best Delay"].values[0])
        return -3  # Default value if stock is not found

    def preprocess_data(self, shifted_df, lookback):
        shifted_df_as_np = shifted_df.to_numpy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(shifted_df_as_np)

        X, y = self.preprocessor.create_sequences(scaled_data)
        X_train, y_train, X_val, y_val, X_test, y_test = self.preprocessor.split_data(X, y)

        X_train, y_train = self.preprocessor.reshape_data(X_train, y_train, lookback)
        X_val, y_val = self.preprocessor.reshape_data(X_val, y_val, lookback)
        X_test, y_test = self.preprocessor.reshape_data(X_test, y_test, lookback)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def create_data_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test):
        train_loader = DataLoader(TimeSeriesDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float()),
                                  batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TimeSeriesDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float()),
                                batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(TimeSeriesDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float()),
                                 batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def initialize_model(self, input_size):
        model = LSTM(input_size, self.hidden_size, self.num_layers, self.dropout_rate).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_function = torch.nn.MSELoss()
        return model, optimizer, loss_function

    def train_with_early_stopping(self, model, train_loader, val_loader, optimizer, loss_function):
        early_stopping = EarlyStopping(patience=self.patience)
        for epoch in range(self.num_epochs_for_stock):
            train_one_epoch(epoch, model, train_loader, optimizer, loss_function, self.device)
            val_loss = validate_one_epoch(model, val_loader, loss_function, self.device)
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
        return val_loss

    def objective(self, trial, stock_name):
        num_epochs_for_stock = trial.suggest_int("num_epochs_for_stock", 50, 300)
        batch_size = trial.suggest_int("batch_size", 16, 128)
        hidden_size = trial.suggest_int("hidden_size", 50, 500)
        num_layers = trial.suggest_int("num_layers", 1, 10)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)

        self.num_epochs_for_stock = num_epochs_for_stock
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        initial_delay = self.get_initial_delay(stock_name)
        shifted_df = StockCorrelationProcessor(self.dataframes).create_time_based_sliding_window(
            n_steps=30,
            stock_name=stock_name,
            date_step='1D',
            n_top=3,
            delay=initial_delay,
            start_date='2016-01-01'
        )
        if shifted_df.empty:
            return float('inf')

        X_train, y_train, X_val, y_val, X_test, y_test = self.preprocess_data(shifted_df, lookback=30)
        train_loader, val_loader, _ = self.create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test)

        model, optimizer, loss_function = self.initialize_model(X_train.shape[2])
        val_loss = self.train_with_early_stopping(model, train_loader, val_loader, optimizer, loss_function)
        return val_loss

    def optimize_for_stock(self, stock_name):
        study = optuna.create_study(direction="minimize")
        study.optimize(partial(self.objective, stock_name=stock_name), n_trials=20)

        best_params = study.best_params
        best_params["stock_name"] = stock_name

        if not hasattr(self, "best_params_df"):
            self.best_params_df = pd.DataFrame(columns=best_params.keys())

        if not self.best_params_df.empty:
            self.best_params_df = pd.concat([self.best_params_df, pd.DataFrame([best_params])], ignore_index=True)
        else:
            self.best_params_df = pd.DataFrame([best_params])
        print(f"Best parameters for {stock_name}: {best_params}")

    def run(self):
        try:
            test_results_df = pd.read_csv(self.best_params_file)
            processed_stocks = set(test_results_df["stock_name"].tolist())
        except FileNotFoundError:
            processed_stocks = set()

        for stock_name in self.dataframes.keys():
            if stock_name not in processed_stocks:
                self.optimize_for_stock(stock_name)
            else:
                print(f"Skipping {stock_name}: Already processed.")

        self.best_params_df.to_csv(self.best_params_file, index=False)
        print(f"Saved best parameters to {self.best_params_file}")


if __name__ == "__main__":
    pipeline = StockTrainingPipelineWithOptuna()
    pipeline.run()

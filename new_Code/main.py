import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from load_all_stocks import StockDataLoader
from StockCorrelationProcessor_GPU import StockCorrelationProcessor
from data_preprocessing import DataPreprocessor, TimeSeriesDataset
from model import LSTM
from training import train_one_epoch, validate_one_epoch, test_model, EarlyStopping
from optimization import mutate_parameters
from results_saver import ResultsSaver


class StockTrainingPipeline:
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
        self.results_saver = ResultsSaver(test_results_file='test_results_new_1.csv')
        self.preprocessor = DataPreprocessor()

        self.loader.load_all_stocks()
        self.dataframes = self.loader.get_dataframes()

        # Load the min error file into a DataFrame
        self.min_error_df = pd.read_csv(self.min_error_file)

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
        """
        Retrieve the initial delay for a stock from the min error file.
        """
        row = self.min_error_df[self.min_error_df["Stock"] == stock_name]
        if not row.empty:
            return int(row["Best Delay"].values[0])
        return -3  # Default value if stock is not found

    def process_stock(self, stock_name):
        print(f"\nProcessing stock: {stock_name}")

        initial_delay = self.get_initial_delay(stock_name)  # Use the new function
        best_params = (30, initial_delay, 3)  # Initial parameters for lookback, delay, n_top
        cache = {}

        best_val_loss = float('inf')
        best_model = None
        best_model_state = None

        for epoch in range(self.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.num_epochs} for stock: {stock_name} ===")

            lookback, delay, n_top = mutate_parameters(best_params)
            print(f"Testing with parameters: Lookback={lookback}, Delay={delay}, N_Top={n_top}")

            cache_key = (lookback, delay, n_top)
            if cache_key in cache:
                shifted_df = cache[cache_key]
            else:
                print(f"Start StockCorrelationProcessor: {stock_name}")
                shifted_df = StockCorrelationProcessor(self.dataframes).create_time_based_sliding_window(
                    n_steps=lookback,
                    stock_name=stock_name,
                    date_step='1D',
                    n_top=n_top,
                    delay=delay,
                    start_date='2016-01-01'
                )
                if shifted_df.empty:
                    print(f"No data available for parameters: {cache_key}")
                    continue
                cache[cache_key] = shifted_df

            X_train, y_train, X_val, y_val, X_test, y_test = self.preprocess_data(shifted_df, lookback)
            train_loader, val_loader, test_loader = self.create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test)

            model, optimizer, loss_function = self.initialize_model(X_train.shape[2]) 
            val_loss = self.train_with_early_stopping(model, train_loader, val_loader, optimizer, loss_function)
          
             # Calculate validation errors (assuming test_model can run on validation set)
            mse, mae, rmse, mape = test_model(model, val_loader, self.device)

            # Save training results for this epoch
            self.results_saver.save_training_results(
                stock_name=stock_name,
                epoch=epoch + 1,
                lookback=lookback,
                delay=delay,
                n_top=n_top,
                val_loss=val_loss,
                mse=mse,
                mae=mae,
                rmse=rmse,
                mape=mape
            )
            # Check if this model is the best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                best_model_state = model.state_dict()
                best_params = (lookback, delay, n_top)

        # Test the best model
        if best_model:
            # Regenerate data based on the best parameters
            lookback, delay, n_top = best_params
            print(f"Regenerating data for testing with parameters: Lookback={lookback}, Delay={delay}, N_Top={n_top}")
            shifted_df = StockCorrelationProcessor(self.dataframes).create_time_based_sliding_window(
                n_steps=lookback,
                stock_name=stock_name,
                date_step='1D',
                n_top=n_top,
                delay=delay,
                start_date='2016-01-01'
            )
            if shifted_df.empty:
                print("No data available for testing with the best parameters.")
            else:
                # Preprocess the data for testing
                X_train, y_train, X_val, y_val, X_test, y_test = self.preprocess_data(shifted_df, lookback)
                train_loader, val_loader, test_loader = self.create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test)

                # Test the model
                best_model.load_state_dict(best_model_state)
                mse, mae, rmse, mape = test_model(best_model, test_loader, self.device)
                print(f"Best model for {stock_name} with parameters {best_params}: MSE={mse}, MAE={mae}, RMSE={rmse}, MAPE={mape}")
                self.results_saver.save_testing_results(
                    stock_name=stock_name,
                    lookback=best_params[0],
                    delay=best_params[1],
                    n_top=best_params[2],
                    mse=mse,
                    mae=mae,
                    rmse=rmse,
                    mape=mape
                )


    def preprocess_data(self, shifted_df, lookback):
        if shifted_df.empty:
            raise ValueError("Shifted dataframe is empty.")

        shifted_df_as_np = shifted_df.to_numpy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(shifted_df_as_np)

        X, y = self.preprocessor.create_sequences(scaled_data)
        if X.shape[0] <= 0 or y.shape[0] <= 0:
            raise ValueError("Insufficient data after creating sequences.")

        if lookback <= 0:
            raise ValueError(f"Invalid lookback value: {lookback}")

        X_train, y_train, X_val, y_val, X_test, y_test = self.preprocessor.split_data(X, y)
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data is empty.")
        if len(X_val) == 0 or len(y_val) == 0:
            raise ValueError("Validation data is empty.")
        if len(X_test) == 0 or len(y_test) == 0:
            raise ValueError("Test data is empty.")

        # Pass `lookback` when calling `reshape_data`
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
            
            if len(val_loader) == 0:
                print("Validation loader is empty. Skipping validation.")
                val_loss = float('inf')
            else:
                val_loss = validate_one_epoch(model, val_loader, loss_function, self.device)
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
        return val_loss

    def run(self):
    # Load the test results CSV file
        try:
            test_results_df = pd.read_csv('test_results_new_1.csv')
            processed_stocks = set(test_results_df['Stock'].tolist())
        except FileNotFoundError:
            print("test_results.csv not found. Processing all stocks.")
            processed_stocks = set()
        
        for stock_name in self.dataframes.keys():
            if (stock_name  not in processed_stocks):
                print(f"Processing stock: {stock_name}")
                self.process_stock(stock_name)
            else:
                print(f"Skipping {stock_name}: Not found in test_results.csv")



if __name__ == "__main__":
    pipeline = StockTrainingPipeline()
    pipeline.run()

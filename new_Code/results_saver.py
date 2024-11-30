import pandas as pd

class ResultsSaver:
    """
    A class to handle saving training and testing results to CSV files.
    """

    def __init__(self, train_results_file="results_new_1.csv", test_results_file="test_results.csv"):
        """
        Initialize the ResultsSaver with file names for saving results.

        Parameters:
            train_results_file (str): File name for training results.
            test_results_file (str): File name for testing results.
        """
        self.train_results_file = train_results_file
        self.test_results_file = test_results_file

    def save_training_results(self, stock_name, epoch, lookback, delay, n_top, val_loss, mse, mae, rmse, mape):
        """
        Save training results to the CSV file.

        Parameters:
            stock_name (str): The stock name.
            epoch (int): The number of training epochs.
            lookback (int): Lookback window size.
            delay (int): Delay for prediction.
            n_top (int): Number of top correlated stocks.
            val_loss (float): Validation loss.
            mse (float): Mean Squared Error.
            mae (float): Mean Absolute Error.
            rmse (float): Root Mean Squared Error.
            mape (float): Mean Absolute Percentage Error.
        """
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
        with open(self.train_results_file, mode='a') as f:
            df.to_csv(f, header=f.tell() == 0, index=False)

    def save_testing_results(self, stock_name, lookback, delay, n_top, mse, mae, rmse, mape):
        """
        Save testing results to the CSV file.

        Parameters:
            stock_name (str): The stock name.
            lookback (int): Lookback window size.
            delay (int): Delay for prediction.
            n_top (int): Number of top correlated stocks.
            mse (float): Mean Squared Error.
            mae (float): Mean Absolute Error.
            rmse (float): Root Mean Squared Error.
            mape (float): Mean Absolute Percentage Error.
        """
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
        with open(self.test_results_file, mode='a') as f:
            df.to_csv(f, header=f.tell() == 0, index=False)

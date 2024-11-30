import os
import pandas as pd
import concurrent.futures

class StockDataLoader:
    def __init__(self, directory_path):
        """
        Initialize the StockDataLoader with directory path and list of stocks.
        """
        self.directory_path = directory_path
        self.dataframes = {}

    def _load_stock_data(self, filename):
        """
        Load individual stock data if it belongs to the list of stocks_server_1.
        """
        stock_name = filename.split('.')[0]
        file_path = os.path.join(self.directory_path, filename)
        return stock_name, pd.read_csv(file_path)

    def load_all_stocks(self):
        """
        Load all stock data concurrently and populate the dataframes dictionary.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda filename: self._load_stock_data(filename),
                os.listdir(self.directory_path)
            )
            self.dataframes = {result[0]: result[1] for result in results if result}

    def get_dataframes(self):
        """
        Return the loaded dataframes.
        """
        return self.dataframes



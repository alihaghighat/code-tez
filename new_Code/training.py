import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np


def train_one_epoch(epoch, model, train_loader, optimizer, loss_function, device, early_stopping=None):
    """
    Train the model for one epoch with optional early stopping.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = loss_function(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss = {loss.item():.4f}")

        # Check early stopping condition after each batch (optional)
        if early_stopping:
            if early_stopping.early_stop:
                print("Early stopping triggered during training!")
                break

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")
    return avg_loss


def validate_one_epoch(model, val_loader, loss_function, device):
    """
    Validate the model for one epoch.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = loss_function(outputs, y_batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def test_model(model, test_loader, device):
    """
    Test the model and calculate evaluation metrics.
    """
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(y_batch.cpu().numpy().flatten())

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actuals, predictions)

    print(f"Test Results - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")
    return mse, mae, rmse, mape


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        Initialize EarlyStopping with patience and minimum delta.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Check if early stopping condition is met.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered!")




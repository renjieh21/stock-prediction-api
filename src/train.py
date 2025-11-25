# src/train.py

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import get_processed_data
from dataset import StockDataset
from model_def import MLPRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
# if torch.backends.mps.is_available():
#     DEVICE = torch.device('mps')
# elif torch.cuda.is_available():
#     DEVICE = torch.device('cuda')
# else:
#     DEVICE = torch.device('cpu')

DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    """
    training loop
    """
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for features, target in train_loader:
            features, target = features.to(DEVICE), target.to(DEVICE)

            outputs = model(features)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        # Model Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, target in val_loader:
                features, target = features.to(DEVICE), target.to(DEVICE)
                outputs = model(features)
                loss = criterion(outputs, target)
                val_loss += loss.item() * features.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        print(f'Epoch {epoch + 1}/{epochs} \t| Train Loss: {train_loss:.6f} \t| Val Loss: {val_loss:.6f}')

        # Saving the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), 'model/model.pth')
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'model.pth'))
            print(f'Model saved to model/model.pth (Val Loss: {val_loss:.6f})')

    print("Training complete.")


def evaluate_model(model, test_loader):
    """
    Evaluate model on test set
    """
    print("Evaluating model on test set...")
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for features, target in test_loader:
            features, target = features.to(DEVICE), target.to(DEVICE)
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(target.cpu().numpy())

    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(f"\nTest Set Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"R-squared (R2 Score): {r2:.6f}")


def main(args):
    data, feature_cols, target_col = get_processed_data(
        args.ticker,
        args.start_date,
        args.end_date,
        args.future_days
    )

    X = data[feature_cols]
    y = data[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # saving scaler and then deploy in API
    # joblib.dump(scaler, 'model/scaler.joblib')
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    print("Scaler saved to model/scaler.joblib")

    # Train Test Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.1, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(0.1 / 0.9), shuffle=False  # (10% / 90%)
    )

    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = len(feature_cols)
    model = MLPRegression(input_dim).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs)

    model.load_state_dict(torch.load('model/model.pth'))
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stock Return Prediction Model")

    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--start_date", type=str, default="2015-01-01", help="Start date")
    parser.add_argument("--end_date", type=str, default="2023-12-31", help="End date")
    parser.add_argument("--future_days", type=int, default=5, help="Days to predict future return")

    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")

    args = parser.parse_args()

    import os

    if not os.path.exists('model'):
        os.makedirs('model')

    main(args)

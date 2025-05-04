import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def load_and_prepare_data():
    """Load and prepare the data for modeling."""
    df = pd.read_csv('../data/Tesla-YTD.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    return df

def train_model(X, y):
    """Train a linear regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

def plot_predictions(X_test, y_test, y_pred, save_path):
    """Create a plot comparing actual vs predicted prices."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')
    plt.title('Tesla Stock Price Prediction using Linear Regression', fontsize=14)
    plt.xlabel('Days from Start', fontsize=12)
    plt.ylabel('Closing Price ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'price_prediction.png'), dpi=300)
    plt.close()

def make_future_predictions(model, df, days_ahead=5):
    """Make predictions for future days."""
    last_day = df['Days'].max()
    future_days = np.array(range(last_day + 1, last_day + days_ahead + 1)).reshape(-1, 1)
    future_predictions = model.predict(future_days)
    return future_predictions

def main():
    # Create reports directory if it doesn't exist
    reports_dir = '../reports'
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load and prepare data
    df = load_and_prepare_data()
    X = df[['Days']]
    y = df['Close']
    
    # Train and evaluate model
    model, X_test, y_test = train_model(X, y)
    mse, r2, y_pred = evaluate_model(model, X_test, y_test)
    
    # Plot predictions
    plot_predictions(X_test, y_test, y_pred, reports_dir)
    
    # Make future predictions
    future_predictions = make_future_predictions(model, df)
    
    # Print results
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")
    print(f"Predicted price increase per day: ${model.coef_[0]:.2f}")
    
    print("\nNext 5 Days Predictions:")
    for i, pred in enumerate(future_predictions, 1):
        print(f"Day {i}: ${pred:.2f}")

if __name__ == "__main__":
    main() 
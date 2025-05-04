import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def load_data():
    """Load and prepare the stock data."""
    df = pd.read_csv('../data/Tesla-YTD.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def create_price_trend_plot(df, save_path):
    """Create a line plot of closing prices over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], linewidth=2)
    plt.title('Tesla Stock Closing Prices (YTD)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Closing Price ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'price_trend.png'), dpi=300)
    plt.close()

def create_volume_price_plot(df, save_path):
    """Create a scatter plot of volume vs price."""
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(df['Volume'], df['Close'], c=df['Close'], cmap='viridis', alpha=0.6)
    plt.title('Trading Volume vs Closing Price', fontsize=14)
    plt.xlabel('Trading Volume', fontsize=12)
    plt.ylabel('Closing Price ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Closing Price ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'volume_price.png'), dpi=300)
    plt.close()

def calculate_statistics(df):
    """Calculate and return basic statistics about the stock data."""
    stats = {
        'average_price': df['Close'].mean(),
        'highest_price': df['Close'].max(),
        'lowest_price': df['Close'].min(),
        'average_volume': df['Volume'].mean()
    }
    return stats

def main():
    # Create reports directory if it doesn't exist
    reports_dir = '../reports'
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Create visualizations
    create_price_trend_plot(df, reports_dir)
    create_volume_price_plot(df, reports_dir)
    
    # Calculate and print statistics
    stats = calculate_statistics(df)
    print("\nTesla Stock Analysis Summary:")
    print(f"Average Closing Price: ${stats['average_price']:.2f}")
    print(f"Highest Closing Price: ${stats['highest_price']:.2f}")
    print(f"Lowest Closing Price: ${stats['lowest_price']:.2f}")
    print(f"Average Daily Volume: {stats['average_volume']:,.0f} shares")

if __name__ == "__main__":
    main() 
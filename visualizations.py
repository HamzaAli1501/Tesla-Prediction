import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the data
df = pd.read_csv('Tesla-YTD.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set the style
plt.style.use('seaborn')
sns.set_palette("husl")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 1. Line plot of closing prices over time
ax1.plot(df['Date'], df['Close'], linewidth=2)
ax1.set_title('Tesla Stock Closing Prices (YTD)', fontsize=14)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Closing Price ($)', fontsize=12)
ax1.grid(True, alpha=0.3)

# 2. Volume vs Price scatter plot
scatter = ax2.scatter(df['Volume'], df['Close'], c=df['Close'], cmap='viridis', alpha=0.6)
ax2.set_title('Trading Volume vs Closing Price', fontsize=14)
ax2.set_xlabel('Trading Volume', fontsize=12)
ax2.set_ylabel('Closing Price ($)', fontsize=12)
ax2.grid(True, alpha=0.3)

# Add colorbar to the scatter plot
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Closing Price ($)')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('tesla_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate and print some basic statistics
print("\nTesla Stock Analysis Summary:")
print(f"Average Closing Price: ${df['Close'].mean():.2f}")
print(f"Highest Closing Price: ${df['Close'].max():.2f}")
print(f"Lowest Closing Price: ${df['Close'].min():.2f}")
print(f"Average Daily Volume: {df['Volume'].mean():,.0f} shares") 
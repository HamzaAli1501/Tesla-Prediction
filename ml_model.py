import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load and prepare the data
df = pd.read_csv('Tesla-YTD.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Prepare features and target
X = df[['Days']]
y = df['Close']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')
plt.title('Tesla Stock Price Prediction using Linear Regression', fontsize=14)
plt.xlabel('Days from Start', fontsize=12)
plt.ylabel('Closing Price ($)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('tesla_prediction.png', dpi=300, bbox_inches='tight')
plt.close()

# Print model performance
print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
print(f"Predicted price increase per day: ${model.coef_[0]:.2f}")

# Predict next 5 days
last_day = df['Days'].max()
future_days = np.array(range(last_day + 1, last_day + 6)).reshape(-1, 1)
future_predictions = model.predict(future_days)

print("\nNext 5 Days Predictions:")
for i, pred in enumerate(future_predictions, 1):
    print(f"Day {i}: ${pred:.2f}") 
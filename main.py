import pandas as pd

# Load the Tesla stock CSV file
file_path = "./Tesla-YTD.csv"
df = pd.read_csv(file_path)

# Show the first few rows and column names to understand structure
df.head(), df.columns.tolist()
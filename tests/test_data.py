import pandas as pd
import os

def test_data_loading():
    """Test if the data file exists and can be loaded correctly."""
    data_path = '../data/Tesla-YTD.csv'
    assert os.path.exists(data_path), "Data file not found"
    
    df = pd.read_csv(data_path)
    assert not df.empty, "DataFrame is empty"
    assert 'Date' in df.columns, "Date column missing"
    assert 'Close' in df.columns, "Close column missing"
    assert 'Volume' in df.columns, "Volume column missing"
    
    print("Data loading test passed successfully!")

if __name__ == "__main__":
    test_data_loading() 
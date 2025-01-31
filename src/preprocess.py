import pandas as pd
import numpy as np

def load_data(filepath):
    """Loads weather dataset and normalizes features."""
    df = pd.read_csv(filepath)
    
    # Selecting relevant columns (e.g., temperature, humidity)
    df = df[["Temperature", "Humidity", "Pressure"]]
    
    # Normalize data
    df = (df - df.min()) / (df.max() - df.min())
    
    return df.values

if __name__ == "__main__":
    data = load_data("data/weather_data.csv")
    print("✅ Data Loaded and Normalized")

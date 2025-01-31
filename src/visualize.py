import matplotlib.pyplot as plt
import pandas as pd

def plot_weather_trends(filepath):
    """Plots weather parameters over time."""
    df = pd.read_csv(filepath)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(3, 1, 1)
    plt.plot(df["Temperature"], label="Temperature", color="red")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(df["Humidity"], label="Humidity", color="blue")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(df["Pressure"], label="Pressure", color="green")
    plt.legend()
    
    plt.suptitle("Weather Trends Over Time")
    plt.show()

if __name__ == "__main__":
    plot_weather_trends("data/weather_data.csv")

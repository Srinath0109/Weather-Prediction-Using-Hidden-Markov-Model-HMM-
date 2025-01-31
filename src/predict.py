import numpy as np
import pickle
from preprocess import load_data

# Load trained HMM model
with open("models/weather_hmm.pkl", "rb") as f:
    model = pickle.load(f)

def predict_weather(observations):
    """Predicts weather states based on past observations."""
    states = model.predict(observations)
    state_map = {0: "Sunny", 1: "Cloudy", 2: "Rainy"}  # Adjust as needed
    return [state_map[state] for state in states]

if __name__ == "__main__":
    test_data = load_data("data/test_weather_data.csv")
    predictions = predict_weather(test_data)
    print("Predicted Weather States:", predictions)

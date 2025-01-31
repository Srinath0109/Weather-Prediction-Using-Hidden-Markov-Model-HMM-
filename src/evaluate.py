import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from predict import predict_weather
from preprocess import load_data

# Load test data
test_data = load_data("data/test_weather_data.csv")

# True weather states (from dataset)
df = pd.read_csv("data/test_weather_data.csv")
true_states = df["WeatherState"].map({"Sunny": 0, "Cloudy": 1, "Rainy": 2}).values

# Predict states
predicted_states = predict_weather(test_data)
predicted_states_numeric = [0 if p == "Sunny" else 1 if p == "Cloudy" else 2 for p in predicted_states]

# Evaluate
print("✅ Accuracy:", accuracy_score(true_states, predicted_states_numeric))
print("📊 Classification Report:\n", classification_report(true_states, predicted_states_numeric))

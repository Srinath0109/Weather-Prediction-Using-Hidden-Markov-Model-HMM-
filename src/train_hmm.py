import numpy as np
from hmmlearn import hmm
from preprocess import load_data
import pickle

# Load weather data
X = load_data("data/weather_data.csv")

# Define HMM model
n_components = 3  # Example: 3 weather states (Sunny, Cloudy, Rainy)
model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000)
model.fit(X)

# Save trained model
with open("models/weather_hmm.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ HMM Model Trained and Saved!")

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
file_path = "crime.csv"  # Ensure this file is inside your project
df = pd.read_csv(file_path)

# Convert 'month' column to datetime
df['month'] = pd.to_datetime(df['month'], format='%d.%m.%Y')
df['Year'] = df['month'].dt.year
df['Month'] = df['month'].dt.month

# Select features and target variable
features = ['Year', 'Month']
target = 'Total_crimes'

X = df[features]
y = df[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a new RandomForestRegressor model (compatible with scikit-learn 1.6.1)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the newly trained model
with open("crime_model.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)

print("✅ Model successfully re-trained and saved as crime_model.pkl (compatible with scikit-learn 1.6.1)")

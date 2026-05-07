import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("data/sales.csv")

# Features
X = df[['advertising', 'holiday', 'temperature', 'customers']]

# Target
y = df['sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model/model.pkl")

print("Model trained successfully!")

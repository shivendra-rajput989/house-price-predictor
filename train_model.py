import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Step 1: Create a dummy dataset
data = {
    'area': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'age': [10, 5, 8, 4, 2],
    'price': [50, 70, 85, 110, 150]  # in lakhs
}

df = pd.DataFrame(data)

# Step 2: Split data
X = df[['area', 'bedrooms', 'age']]
y = df['price']

# Step 3: Train model
model = LinearRegression()
model.fit(X, y)

# Step 4: Save trained model
joblib.dump(model, 'house_price_model.pkl')

print("✅ Model trained and saved as house_price_model.pkl")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Load dataset
df = pd.read_csv("./house_data.csv")
print(df.columns)



# Select features and target variable
df.columns = df.columns.str.strip()  # Removes leading/trailing spaces
X = df[['square_footage', 'bedrooms', 'bathrooms']].values
X = df[['square_footage', 'bedrooms', 'bathrooms']].values
y = df['price'].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Define Linear Regression Model (from scratch)
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias
            error = y_predicted - y

            # Gradient Descent Optimization
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train Model
model = LinearRegressionScratch(learning_rate=0.01, epochs=2000)
model.fit(X, y)

# Make Predictions
y_pred = model.predict(X)
print("Predicted House Prices:", y_pred)

# Visualize Results
plt.scatter(y, y_pred, alpha=0.6, color='blue', label='Predicted vs Actual')
plt.plot([min(y), max(y)], [min(y), max(y)], '--', color='red', label='Perfect Fit')

plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()

# Save Model for Future Use
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Get User Input for House Features
try:
    square_footage = float(input("Enter square footage of the house: "))
    bedrooms = int(input("Enter number of bedrooms: "))
    bathrooms = int(input("Enter number of bathrooms: "))
except ValueError:
    print("❌ Invalid input! Please enter numeric values.")
    exit()

# Convert user input into NumPy array
new_data = np.array([[square_footage, bedrooms, bathrooms]])

# Normalize new data (if training data was normalized)
new_data = (new_data - X.mean(axis=0)) / X.std(axis=0)

# Predict house price using trained model
predicted_price = model.predict(new_data)

# Display the estimated price
print(f"🏡 Estimated Price: ₹{predicted_price[0]:,.2f}")
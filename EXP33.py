import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\kbala\Downloads\cars_data.csv")

print(df.head())

selected_features = ['engine_size', 'horsepower', 'fuel_efficiency']
X = df[selected_features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()

coefficients = model.coef_
intercept = model.intercept_
print(f"\nModel Coefficients: {dict(zip(selected_features, coefficients))}")
print(f"Intercept: {intercept}")


# data set
engine_size  horsepower  fuel_efficiency  price
2.0          150          25              30000
2.5          200          20              45000
2.4          180          22              38000
2.0          120          30              28000
2.2          160          28              35000

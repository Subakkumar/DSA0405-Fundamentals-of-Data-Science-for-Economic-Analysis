import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\kbala\Downloads\house_data.csv")

feature = df[['sqft']]
target = df['price']

plt.scatter(feature, target)
plt.xlabel('House SQFT')
plt.ylabel('House Price')
plt.title('Bivariate Analysis: House SQFT vs. Price')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.scatter(X_test, y_test, label='Actual Prices')
plt.scatter(X_test, y_pred, label='Predicted Prices')
plt.xlabel('House SQFT')
plt.ylabel('House Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()

coefficients = pd.DataFrame({'Feature': feature.columns, 'Coefficient': model.coef_})
print(coefficients)

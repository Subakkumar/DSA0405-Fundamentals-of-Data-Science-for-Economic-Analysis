import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = {
    'age': [25, 30, 35, 40, 45, 28, 32, 38, 42, 48],
    'income': [50000, 60000, 70000, 80000, 90000, 55000, 65000, 75000, 85000, 95000],
    'browsing_duration': [15, 20, 25, 30, 35, 18, 22, 28, 33, 40],
    'device_type': ['Mobile', 'Desktop', 'Mobile', 'Desktop', 'Mobile', 'Desktop', 'Mobile', 'Desktop', 'Mobile', 'Desktop'],
    'purchase': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
data = pd.DataFrame(df)

print(data.head())

categorical_features = ['device_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough' 
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

X = data.drop('purchase', axis=1)
y = data['purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

new_customer_data = pd.DataFrame({
    'age': [30],
    'income': [50000],
    'browsing_duration': [15],
    'device_type': ['Mobile']
})

prediction = pipeline.predict(new_customer_data)

print("Prediction for new customer:", prediction)

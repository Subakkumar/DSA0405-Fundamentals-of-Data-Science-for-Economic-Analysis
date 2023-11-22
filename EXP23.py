# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.DataFrame({
    'income': [50000, 80000, 60000, 75000, 90000, 55000, 72000, 85000],
    'credit_score': [600, 750, 700, 720, 780, 620, 700, 760],
    'debt_to_income_ratio': [0.4, 0.2, 0.3, 0.2, 0.1, 0.5, 0.3, 0.2],
    'employment_duration': [1, 3, 2, 2, 4, 1, 2, 3],
    'risk': ['high', 'low', 'low', 'low', 'low', 'high', 'low', 'low']
})
print(data.head())
categorical_features = ['employment_duration']

X = data.drop('risk', axis=1)
y = data['risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', 'passthrough', categorical_features),
        ('num', StandardScaler(), X.columns.difference(categorical_features))
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

new_applicant_data = pd.DataFrame({
    'income': [50000],
    'credit_score': [700],
    'debt_to_income_ratio': [0.4],
    'employment_duration': [5]
})
new_applicant_prediction = pipeline.predict(new_applicant_data)

print("\nPrediction for the new loan applicant:", new_applicant_prediction[0])

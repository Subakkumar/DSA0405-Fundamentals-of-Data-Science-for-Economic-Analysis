import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv(r"C:\Users\kbala\Downloads\medical_data.csv")

print(df.head())

features = ['age', 'gender', 'blood_pressure', 'cholesterol']
X = df[features]
y = df['treatment_outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'blood_pressure', 'cholesterol']),
        ('cat', OneHotEncoder(), ['gender'])
    ])

k_value = 3
knn_model = KNeighborsClassifier(n_neighbors=k_value)
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', knn_model)
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Good')
recall = recall_score(y_test, y_pred, pos_label='Good')
f1 = f1_score(y_test, y_pred, pos_label='Good')
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

print("\nConfusion Matrix:")
print(conf_matrix)


# data set
age,gender,blood_pressure,cholesterol,treatment_outcome
45,Male,120,200,Good
60,Female,140,220,Bad
35,Female,130,180,Good
50,Male,135,240,Bad
40,Female,125,190,Good
55,Male,128,210,Good

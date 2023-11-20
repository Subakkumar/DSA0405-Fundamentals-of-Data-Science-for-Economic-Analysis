import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
medical_data = pd.read_csv(r"C:\Users\kbala\Downloads\med_data.csv")
medical_data = pd.get_dummies(medical_data, columns=['Gender'], drop_first=True)
X = medical_data.drop('Outcome', axis=1)
y = medical_data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Good')
recall = recall_score(y_test, y_pred, pos_label='Good')
f1 = f1_score(y_test, y_pred, pos_label='Good')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')

conf_matrix = confusion_matrix(y_test, y_pred, labels=['Good', 'Bad'])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
print(medical_data)


#data set
age	Gender	blood_Pressure	cholesterol_level	Outcome
18	M	      39	            13	              Good
19	F	      40	            20	              Good
23	M	      36	            20	              Bad
24	M	      34	            23	              Good
27	F	      55	            36	              Good
40	F	      43	            39	              Bad
36	M	      46	            34	              Bad
34	M	      58	            23	              Good
55	M	      45	            25	              Bad
57	M	      45	            23	              Good
58	M	      45	            24	              Good

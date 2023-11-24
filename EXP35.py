import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv(r"C:\Users\kbala\Downloads\customer_data.csv")

print(df.head())

X = df[['amount_spent', 'frequency_of_visits']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

num_clusters = 3  
kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)

pipeline = make_pipeline(scaler, kmeans_model)

pipeline.fit(X)

df['cluster'] = kmeans_model.labels_

print("\nCluster Assignments:")
print(df[['customer_id', 'cluster']])

plt.scatter(X['amount_spent'], X['frequency_of_visits'], c=df['cluster'], cmap='viridis')
plt.xlabel('Amount Spent')
plt.ylabel('Frequency of Visits')
plt.title('Customer Segmentation')
plt.show()

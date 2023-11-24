import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

data = {'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'TotalAmountSpent': [100, 150, 80, 200, 120, 180, 250, 300, 150, 200],
        'NumItemsPurchased': [5, 8, 3, 10, 6, 9, 12, 15, 8, 10]}

df = pd.DataFrame(data)

print(df.head())

X = df[['TotalAmountSpent', 'NumItemsPurchased']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

num_clusters = 3  
kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)

pipeline = make_pipeline(scaler, kmeans_model)

pipeline.fit(X)

df['Cluster'] = kmeans_model.labels_

print("\nCluster Assignments:")
print(df[['CustomerID', 'Cluster']])

plt.figure(figsize=(8, 6))
for cluster in range(num_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['TotalAmountSpent'], cluster_data['NumItemsPurchased'], label=f'Cluster {cluster + 1}')

plt.title('Customer Segmentation based on Spending and Purchase Behavior')
plt.xlabel('Total Amount Spent')
plt.ylabel('Number of Items Purchased')
plt.legend()
plt.show()

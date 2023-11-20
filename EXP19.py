import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    'CustomerID' : [1,2,3,4,3,2,4,1,3,2],
    'TotalAmount' : [200,100,230,320,400,900,100,140,150,800],
    'NumItemsPurchased' : [3,5,2,5,6,7,8,2,1,5]
    }

transaction_data = pd.DataFrame(data)

features = ['TotalAmount', 'NumItemsPurchased']
X = transaction_data[features]

wcss = []  

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster-Sum-of-Squares (WCSS)')
plt.show()

optimal_k = 3  

kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans_model.fit_predict(X)

transaction_data['Cluster'] = clusters

plt.scatter(transaction_data['TotalAmount'], transaction_data['NumItemsPurchased'], c=clusters, cmap='viridis', edgecolors='k', s=50)
plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Customer Segmentation based on Spending and Purchase Behavior')
plt.xlabel('Total Amount Spent')
plt.ylabel('Number of Items Purchased')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'transaction_data' with columns: 'CustomerID', 'TotalAmount', 'Frequency'
data = {
    'CustomerID' : [1,2,3,4,5,6,7,8,9,10],
    'TotalAmount' : [2000,3000,1000,1500,3200,4000,2500,9000,2300,8000],
    'Frequency' : [3,2,5,6,8,2,3,5,9,5]
    }
# Load or preprocess your transaction data here
transaction_data=pd.DataFrame(data)
# Select relevant features for clustering
features = ['TotalAmount', 'Frequency']
X = transaction_data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-Cluster-Sum-of-Squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster-Sum-of-Squares (WCSS)')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters
optimal_k = 3  # Adjust this based on the plot

# Train the K-Means model with the optimal number of clusters
kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans_model.fit_predict(X_scaled)

# Add cluster labels to the original DataFrame
transaction_data['Cluster'] = clusters

# Display the cluster centers
cluster_centers = scaler.inverse_transform(kmeans_model.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
print("Cluster Centers:")
print(cluster_centers_df)

# Visualize the clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', edgecolors='k', s=50)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Customer Segmentation based on Spending Patterns')
plt.xlabel('Total Amount Spent')
plt.ylabel('Frequency of Visits')
plt.legend()
plt.show()

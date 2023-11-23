import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = {
    'PurchaseFrequency': [2, 5, 1, 6, 10, 2, 8, 3, 9, 7],
    'TimeSpentOnSite': [10, 20, 5, 25, 30, 15, 35, 10, 40, 25],
    'Age': [25, 35, 22, 45, 50, 30, 55, 28, 60, 40]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

k = 3

kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['Cluster'] = kmeans.fit_predict(scaled_data)

centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=df.columns[:-1])
print("\nCentroid values for each cluster:")
print(centroid_df)

plt.figure(figsize=(10, 6))
for cluster in range(k):
    plt.scatter(df[df['Cluster'] == cluster]['TimeSpentOnSite'], df[df['Cluster'] == cluster]['PurchaseFrequency'], label=f'Cluster {cluster}')

plt.scatter(centroids[:, 1], centroids[:, 0], marker='X', s=200, color='red', label='Centroids')
plt.title('Customer Segmentation')
plt.xlabel('Time Spent On Site')
plt.ylabel('Purchase Frequency')
plt.legend()
plt.show()

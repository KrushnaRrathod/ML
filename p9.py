import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the training dataset from a CSV file
df = pd.read_csv(r'C:\Users\manan\OneDrive\Desktop\6 sem degree\ML\large_dataset.csv')

# Extract features from the dataset
X = df.iloc[:, [0, 1]].values  # Assuming the dataset has two features for simplicity

# Initialize the KMeans model with the desired number of clusters (k)
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Fit the KMeans model to the dataset
kmeans.fit(X)

# Visualize the clusters
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='*', label='Centroids')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

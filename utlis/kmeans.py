from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def calculate_kmeans(df, title, values):
    features = df.pivot(index='Country Name', columns='Indicator Name', values = values)

    kmeans = KMeans(n_clusters=10, init='k-means++', n_init=10, random_state=42)

    kmeans.fit(features)

    features['Cluster'] = kmeans.labels_

    mean_values = features.mean(axis=1)

    plt.figure(figsize=(18, 6))
    features_sorted = features.sort_values('Cluster')
    plt.scatter(features_sorted.index, mean_values, c=features_sorted['Cluster'], cmap='viridis')
    plt.xlabel('Country Name')
    plt.ylabel('Mean Value')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns = features.columns.values[:-1])

    return (features, centroids)
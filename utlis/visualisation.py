import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import umap
import umap.plot
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_dendogram(df, title, values):
    features = df.pivot(index='Country Name', columns='Indicator Name', values=values)
    distance_matrix = hierarchy.distance.pdist(features)

    dendrogram = hierarchy.linkage(distance_matrix, method='complete')

    plt.figure(figsize=(15, 20))
    hierarchy.dendrogram(dendrogram, labels=features.index, leaf_rotation=00, orientation='right')
    plt.xlabel('Distance')
    plt.ylabel('Country Name')
    plt.title(f'Dendrogram: {title}')
    plt.show()

def plot_tsne(df, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(df)
    plot_scatter(tsne_features, labels, df.index)
    plt.xlabel('T-SNE Component 1')
    plt.ylabel('T-SNE Component 2')
    plt.title(f'T-SNE: {title}')
    plt.colorbar(label='Cluster')
    plt.show()

def plot_umap(df, labels, title):
    umap_obj = umap.UMAP(random_state=42)
    umap_features = umap_obj.fit_transform(df)
    plot_scatter(umap_features, labels, df.index)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title(f'U-MAP: {title}')
    plt.colorbar(label='Cluster')
    plt.show()

def plot_pca(df, labels, columns, title, scale = 1):
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(df)
    plot_scatter(pca_features, labels, df.index)

    coef = pca.components_[0:2, :].T

    for label, v in zip(columns, coef):
        cords = (v[0] * scale, v[1] * scale)
        plt.arrow(0, 0, *cords, color='red')
        plt.annotate(label, cords, size=8)

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'PCA: {title}')
    plt.colorbar(label='Cluster')
    plt.show()

def plot_centroid(df, df_with_cluster, title_main):
    fig, ax = plt.subplots(5, 2, figsize=(13, 15))
    fig.tight_layout(pad=5.0)
    for i, centroid in df.iterrows():
        indexes = [str[0:10] for str in centroid.index]
        sns.barplot(x = indexes, y = centroid.values, hue = indexes, ax = ax[i // 2][i % 2])
        title = df_with_cluster[df_with_cluster['Cluster'] == i].index.values.tolist()
        ax[i // 2][i % 2].title.set_text(f"Cluster {i + 1} - {', '.join(title)[0:30]}...")
    fig.suptitle(f'Cetnroids summary: {title_main}')
    fig.show()

def plot_centroid_col(df, df_with_cluster, title_main):
    fig, ax = plt.subplots(10, 1, figsize=(13, 15))
    fig.tight_layout(pad=5.0)
    for i, centroid in df.iterrows():
        indexes = [str[0:5] for str in centroid.index]
        sns.barplot(x = indexes, y = centroid.values, hue = indexes, ax = ax[i])
        title = df_with_cluster[df_with_cluster['Cluster'] == i].index.values.tolist()
        ax[i].title.set_text(f"Cluster {i + 1} - {', '.join(title)[0:30]}...")
    fig.suptitle(f'Cetnroids summary: {title_main}')
    fig.show()

def plot_scatter(features, labels, index):
    plt.figure(figsize=(12, 12))
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')
    for i, label in enumerate(index):
        plt.annotate(label, (features[i, 0], features[i, 1]), size=8)

# ---------------------------------------------- K Means ---------------------------------------------------------

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

# Get dataset and plot 
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:,0], X[:,1])

# determining Value for n_clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Using n_clusters value derived from above elbow method
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()


# ---------------------------------------------- Agglomerative Hierarchical Clustering -----------------------------------------------

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

dataset = pd.read_csv('./data.csv')

X = dataset.iloc[:, [3, 4]].values

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
model.fit(X)
labels = model.labels_

plt.scatter(X[labels==0, 0], X[labels==0, 1], s=50, marker='o', color='red')
plt.scatter(X[labels==1, 0], X[labels==1, 1], s=50, marker='o', color='blue')
plt.scatter(X[labels==2, 0], X[labels==2, 1], s=50, marker='o', color='green')
plt.scatter(X[labels==3, 0], X[labels==3, 1], s=50, marker='o', color='purple')
plt.scatter(X[labels==4, 0], X[labels==4, 1], s=50, marker='o', color='orange')
plt.show()



# ---------------------------------------------- Gaussian Mixture Model -----------------------------------------------






# ---------------------------------------------- t-Distributed Stochastic Neighbor Embedding (t-SNE)  ------------------------------------------

# t-Distributed Stochastic Neighbor Embedding (t-SNE) is a dimensionality reduction technique used to represent high-dimensional dataset in a low-dimensional space of two or 
# three dimensions so that we can visualize it. In contrast to other dimensionality reduction algorithms like PCA which simply maximizes the variance, t-SNE creates a reduced 
# feature space where similar samples are modeled by nearby points and dissimilar samples are modeled by distant points with high probability.
# At a high level, t-SNE constructs a probability distribution for the high-dimensional samples in such a way that similar samples have a high likelihood of being picked 
# while dissimilar points have an extremely small likelihood of being picked. Then, t-SNE defines a similar distribution for the points in the low-dimensional embedding. 
# Finally, t-SNE minimizes the Kullback–Leibler divergence between the two distributions with respect to the locations of the points in the embedding.

# The KL divergence is a measure of how different one probability distribution from a second. e.g. similar distribution KL(P||Q) = 0, nearly similar = 1, 
# different dietribution=3 (i.e. more)

# t-SNE [1] is a tool to visualize high-dimensional data. It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence 
# between the joint probabilities of the low-dimensional embedding and the high-dimensional data. t-SNE has a cost function that is not convex, i.e. with different initializations 
# we can get different results.

# It is highly recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the number of dimensions to a reasonable
# amount (e.g. 50) if the number of features is very high. This will suppress some noise and speed up the computation of pairwise distances between samples.

# Import all library
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

# Load the dataset
X, y = load_digits(return_X_y=True)

# Load the t-SNE and fit the data
tsne = TSNE(n_components=2, perplexity=500.0, n_iter=5000, random_state=0)
X_embedded = tsne.fit_transform(X)

# Hyperparameter:
# ===================
    # n_components=2,
    # perplexity=30.0,
    # early_exaggeration=12.0,
    # learning_rate=200.0,
    # n_iter=1000,
    # n_iter_without_progress=300,
    # min_grad_norm=1e-07,
    # metric='euclidean',
    # init='random',
    # verbose=0,
    # random_state=None,
    # method='barnes_hut',
    # angle=0.5,
    # n_jobs=None)

    
# Plot the data distribution
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full')





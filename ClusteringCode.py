
# There are four type of clustering algorithm-

# https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/


# 1. Connectivity models: 
# These models are based on the notion that the data points closer in data space exhibit more similarity to each other than the data points lying farther away. These models are very easy to interpret but lacks scalability for handling big datasets. eg. hierarchical clustering algorithm and its variant

# 2. Centroid models: These are iterative clustering algorithms in which the notion of similarity is derived by the closeness of a data point to the centroid of the clusters.  e.g. K-means clustering

# 3. Density Models: These models search the data space for areas of varied density of data points in the data space. It isolates various different density regions and assign the data points within these regions in the same cluster. Popular examples of density models are DBSCAN and OPTICS.

# 4. Distribution models: These clustering models are based on the notion of how probable is it that all data points in the cluster belong to the same distribution (For example: Normal, Gaussian). These models often suffer from overfitting. A popular example of these models is Expectation-maximization algorithm which uses multivariate normal distributions.

# ----------------- Difference between K Means and Hierarchical clustering ------------------------------------------

# Hierarchical clustering can’t handle big data well but K Means clustering can. This is because the time complexity of K Means is linear i.e. O(n) while that of hierarchical clustering is quadratic i.e. O(n2).

# In K Means clustering, since we start with random choice of clusters, the results produced by running the algorithm multiple times might differ. While results are reproducible in Hierarchical clustering.

# K Means is found to work well when the shape of the clusters is hyper spherical (like circle in 2D, sphere in 3D).

# K Means clustering requires prior knowledge of K i.e. no. of clusters you want to divide your data into. But, you can stop at whatever number of clusters you find appropriate in hierarchical clustering by interpreting the dendrogram

# ---------------------------------------------- K Means ---------------------------------------------------------

# https://github.com/codebasics/py/blob/master/ML/13_kmeans/13_kmeans_tutorial.ipynb

# Draw back of K-Means 
# https://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_csv('Customers.csv')
# Get Annual Income (k$) and Spending Score (1-100) field
X=df.iloc[:,[3,4]].values

# Plot the original data
plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'])
plt.title('Clusters of customers')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.show()

# Elbow method to find out K value
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


#Fitting K-Means to the dataset
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(X)


#Visualize the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Cluster1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Cluster3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Cluster4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Cluster5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')

plt.title('Clusters of customers')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()

# ---------------------------------------------- Hierarchical Clustering -----------------------------------------------

# https://www.geeksforgeeks.org/ml-hierarchical-clustering-agglomerative-and-divisive-clustering/

# Hierarchical Clustering -
# 1. Agglomerative Hierarchical Clustering
# 2. Divisive hierarchical clustering

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


# ---------------------------------------------- Association Rule Mining via Apriori Algorithm  ------------------------------------------

# https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/

# Association rule mining is a technique to identify underlying relations between different items. Take an example of a Super Market where customers 
# can buy variety of items. Usually, there is a pattern in what the customers buy. For instance, mothers with babies buy baby products such as milk 
# and diapers. Damsels may buy makeup items whereas bachelors may buy beers and chips etc. In short, transactions involve a pattern. 
# More profit can be generated if the relationship between the items purchased in different transactions can be identified.

# For instance, if item A and B are bought together more frequently then several steps can be taken to increase the profit. For example:
# 
# 1. A and B can be placed together so that when a customer buys one of the product he doesn't have to go far away to buy the other product.
# 2. People who buy one of the products can be targeted through an advertisement campaign to buy the other.
# 3. Collective discounts can be offered on these products if the customer buys both of them.
# 4. Both A and B can be packaged together. The process of identifying an associations between products is called association rule mining.

#There are three major components of Apriori algorithm:

# 1. Support
# 2. Confidence
# 3. Lift

# Support refers to the default popularity of an item and can be calculated by finding number of transactions containing a particular item divided by total number of transactions. 
# Support(B) = (Transactions containing (B))/(Total Transactions)

# Confidence refers to the likelihood that an item B is also bought if item A is bought. It can be calculated by finding the number of transactions where A and B are bought together, divided by total number of transactions where A is bought. 

# Confidence(A→B) = (Transactions containing both (A and B))/(Transactions containing A)

# Lift(A→B) = (Confidence (A→B))/(Support (B))

# A Lift of 1 means there is no association between products A and B. Lift of greater than 1 means products A and B are more likely to be bought together. Finally, Lift of less than 1 refers to the case where two products are unlikely to be bought together.

# !pip install apyori

from apyori import apriori
rules = apriori(l,  min_support = 0.1, min_confidence=0.35, min_lift=1.1, max_length=2)
list(rules)


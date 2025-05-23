import pandas as pd
import random
import numpy as np

df = pd.read_csv("kmeans.csv", index_col='no')

def euclidean(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def new_centroids(clusters):
    return [tuple(np.mean(cluster, axis=0)) for cluster in clusters]

k = int(input("Enter no of clusters: "))

c_index = []
while len(c_index) < k:
    x = random.randint(1, 10)
    if x not in c_index:
        c_index.append(x)

centroids = [tuple(df.loc[i]) for i in c_index]
points = list(zip(df['height'], df['weight']))

while True:
    clusters = [[] for _ in range(k)]
    for point in points:
        distances = [euclidean(point, centroid) for centroid in centroids]
        clusters[distances.index(min(distances))].append(point)
    new_centroids_list = new_centroids(clusters)
    if new_centroids_list == centroids:
        break
    centroids = new_centroids_list

print(clusters)

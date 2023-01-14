from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from scipy.spatial import Voronoi, voronoi_plot_2d

import matplotlib.pyplot as plt

import numpy as np


points = []

with open("DS1.txt") as f:
    coordinates = f.readline()

    while coordinates:
        x = int((coordinates.split()[0]))
        y = int(coordinates.split()[1])
        points.append((x, y))
        coordinates = f.readline()



### ESTIMATING epsilon

# neighbors = NearestNeighbors(n_neighbors=4)
# neighbors_fit = neighbors.fit(points_test)
# distances, indices = neighbors_fit.kneighbors(points_test)

# distances = np.sort(distances, axis=0)
# distances = distances[:,1]
# plt.plot(distances)
# plt.show()


db = DBSCAN(eps=4, min_samples=5).fit(points)
labels = db.labels_

### FINDING CENTROIDS

centroids = [0 for i in range(len(set(labels)))]

for j in range(len(set(labels))):
    centroids[j] = np.mean([points[i] for i in range(len(points)) if labels[i] == j], axis=0)


vor = Voronoi(centroids)


fig, ax = plt.subplots()
fig.set_size_inches(9.6, 5.4)
ax.scatter([i[0] for i in points], [i[1] for i in points], color="black", alpha=0.1)
voronoi_plot_2d(vor, ax, line_colors='green', line_width=1.7, line_alpha=1, point_size=10, show_vertices=False)

ax.set_xlim(103.4, 513)
ax.set_ylim(101, 850)

plt.axis("off")

plt.savefig("diagram.png")
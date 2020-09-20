import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy


def drawClusters(clusters, centers):

    clusterX = []
    clusterY = []

    for x in range(len(clusters)):

        clusterX.append([])
        clusterY.append([])

        for y in range(len(clusters[x])):
            clusterX[x].append(clusters[x][y][0])
            clusterY[x].append(clusters[x][y][1])

        plt.scatter(clusterX[x], clusterY[x])
        plt.scatter([centers[x][0]], [centers[x][1]], marker='X', s=20 * 4)

    plt.show()

def euclideanDistance(one, two):
    squared_distance = 0

    # Assuming correct input to the function where the lengths of two features are the same

    for i in range(len(one)):
        squared_distance += (one[i] - two[i]) ** 2

    ed = math.sqrt(squared_distance)

    return ed;

def mykmeans(X, k, c, tolerance=0.0001, max_iterations=10000):

    if (len(c) != k):
        exit("K and number of centroids not matching")
    else:

        # iterate 10000 times
        for x in range(max_iterations):

            clusters = []

            # create 2d list for nodes in clusters
            for i in range(len(c)):
                clusters.append([])

            # categorise data points into their clusters
            for i in range(len(X)):

                # Compute distance between data points and centroids
                tempDistance = 100000
                index = 0

                for j in range(len(c)):
                    dist = euclideanDistance(X[i], c[j])
                    if dist < tempDistance:
                        tempDistance = dist
                        index = j

                clusters[index].append(X[i])

            oldCentroids = deepcopy(c)

            # Update centroids with avg x,y values of each cluster nodes
            for i in range(len(clusters)):

                totalX = 0
                totalY = 0
                for x in range(len(clusters[i])):
                    totalX = totalX + float(clusters[i][x][0])
                    totalY = totalY + float(clusters[i][x][1])

                if len(clusters[i]) != 0:
                    newX = totalX / len(clusters[i])
                    newY = totalY / len(clusters[i])
                    c[i] = [newX, newY]

            # Check threshold with old value
            if x != 0:
                toleranceCheck = 0
                for i in range(len(c)):
                    if euclideanDistance(oldCentroids[i], c[i]) <= tolerance:
                        toleranceCheck += 1

                if toleranceCheck == len(c):
                    return clusters, c, x

    return clusters, c, max_iterations


def createData(mu_list, sigma, N):
    sample = []
    for mu in mu_list:
        sample.append(np.random.multivariate_normal(mu, sigma, N))

    X = np.concatenate((sample[0], sample[1], sample[2]))

    return X


if __name__ == "__main__":

    mu_list = [[-3,0],[3,0],[0,3]]
    sigma = np.array([[1,0.75],[0.75,1]])
    N = 10

    X = createData(mu_list, sigma, N)
    plt.scatter(X[0:10][:,0], X[0:10][:,1])
    plt.scatter(X[10:20][:,0], X[10:20][:,1])
    plt.scatter(X[20:30][:,0], X[20:30][:,1])
    plt.show()

    c = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    clusters, centers, iterations = mykmeans(X, 4, c)
    drawClusters(clusters, centers)

import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy
import secrets
from itertools import permutations


def drawClusters(plotnumber, clusters, centers, k, mu):

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


    plt.savefig(f'{plotnumber}')
    plt.show()


def euclideanDistance(one, two):
    squared_distance = 0

    # Assuming correct input to the function where the lengths of two features are the same

    for i in range(len(one)):
        squared_distance += (one[i] - two[i]) ** 2

    ed = math.sqrt(squared_distance)

    return ed;


def mykmeans(X, k, tolerance=0.0001, max_iterations=10000):

    c = []

    # create random initial centers
    for _ in range(k):
        pick = list(secrets.choice(X))
        while pick in c:
            pick = list(secrets.choice(X))

        c.append(pick)

    print(f"Random points as initial centers = {[[round(val, 2) for val in center] for center in c]}")

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
    l = []
    for i, mu in enumerate(mu_list):
        sample.append(np.random.multivariate_normal(mu, sigma, N))
        l.append(np.zeros(N) + i)

    X = np.concatenate((sample[0], sample[1], sample[2]))
    label = np.concatenate((l[0], l[1], l[2]))

    return X, label


def getAccuracy(X, y, clusters):


    combinations = list(permutations([0, 1, 2]))

    accuracies = []

    for combination in combinations:

        labels = np.zeros(len(X))

        for label, cluster in enumerate(clusters):
            for point in cluster: # [0,1,2]
                index = np.where(X == point)[0][0] # 8
                labels[index] = combination[label]

        accuracies.append(round(len([labels[i] for i in range(0, len(labels)) if labels[i] == y[i]]) / len(labels) * 100, 2))

    return max(accuracies)


if __name__ == "__main__":

    plotnumber = 1

    # PARAMETERS
    mu_lists = [[[-3, 0], [3, 0], [0, 3]], [[-2, 0], [2, 0], [0, 2]]]
    sigma = np.array([[1,0.75],[0.75,1]])
    k_list = [2,3,4,5]
    N = 300

    for mu_list in mu_lists:

        X, label = createData(mu_list, sigma, N)
        i = 0
        plt.scatter(X[i:i + N][:, 0], X[i:i + N][:, 1])
        i = i + N
        plt.scatter(X[i:i + N][:, 0], X[i:i + N][:, 1])
        i = i + N
        plt.scatter(X[i:i + N][:, 0], X[i:i + N][:, 1])
        plt.savefig(f'{plotnumber}')
        plotnumber += 1
        plt.show()

        for k in k_list:

            clusters, centers, iterations = mykmeans(X, k)
            drawClusters(plotnumber, clusters, centers, k , mu_list)
            plotnumber += 1

            if k == 3:
                accuracy = getAccuracy(X, label, clusters)
                print (f'accuracy = {accuracy}%')

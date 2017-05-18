import numpy as np
import sys
import os
from nltk.cluster.kmeans import KMeansClusterer
import random

def negIoU(x, y):
    xmax = min(x[0], y[0])
    ymax = min(x[1], y[1])
    i = xmax * ymax
    u = x[0] * x[1] + y[0] * y[1] - i
    return 1 - i / u
    # xmax = max(x[0]-x[2]/2., y[0]-y[2]/2.)
    # ymax = max(x[1]-x[3]/2., y[1]-y[3]/2.)
    # xmin = min(x[0]+x[2]/2., y[0]+y[2]/2.)
    # ymin = min(x[1]+x[3]/2., y[1]+y[3]/2.)
    # i = (xmin - xmax) * (ymin - ymax)
    # u = x[2] * x[3] + y[2] * y[3] - i;
    # return 1 - i / u

if __name__  == '__main__':
    workspace = os.getcwd() + '/'
    datacfg = workspace + sys.argv[1]
    lines = open(datacfg).readlines()
    images = []
    for line in lines:
        if (line.split(' ')[0] == 'train'):
            valid_path = line.strip().split(' ')[-1]
            if (valid_path[0] != '/'):
                valid_path = workspace + valid_path
            lists = open(valid_path).readlines()
            images = [x.strip() for x in lists]

    bboxes = []
    for image in images:
        label = image.replace('.jpg', '.txt')
        lines = open(label).readlines()
        for line in lines:
            splitline = line.split(' ')
            # bboxes.append([float(x)*13. for x in splitline[-2:]])
            bboxes.append([float(splitline[-2])*1., float(splitline[-1])*1.])
    print(len(bboxes))
    # samples = random.sample(bboxes, 15000)
    # print(len(samples))
    bboxes = np.array(bboxes)
    # samples = np.array(samples)
    # print(samples.shape)

    KMeans = KMeansClusterer(5, negIoU, repeats=1)
    # clusters = KMeans.cluster(samples, True)
    clusters = KMeans.cluster(bboxes, True)
    centroids = KMeans.means()
    print(np.array(centroids) / np.array((1., 1.)))

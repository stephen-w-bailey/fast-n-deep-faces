import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.cluster import KMeans
import sys
import yaml

basePath = (Path(__file__).parent / '..').resolve()

def pdist2(x,y): # x is ...,m,d and y is ...,n,d
    rank = len(x.shape)
    transpose = [i for i in range(rank)]
    transpose[-1] = rank-2
    transpose[-2] = rank-1
    dist = -2*np.matmul(x,np.transpose(y,transpose)) # ...,m,n
    xx = np.sum(x**2,-1) # ...,m
    yy = np.sum(y**2,-1) # ...,n
    dist = dist + xx[...,np.newaxis] + np.expand_dims(yy,-2) # ...,m,n
    return np.maximum(dist,0)

def maskPoints(points,weight=None,k=3):
    kmeans = KMeans(n_clusters=k)
    fullPoints = points
    if weight is not None:
        points = points[weight>0]
        weight = weight[weight>0]
    kmeans.fit(points,sample_weight=weight)
    score = kmeans.score(points,sample_weight=weight)
    centers = np.asarray(kmeans.cluster_centers_)
    meanDist = 0.15
    p = kmeans.predict(fullPoints)
    masks = []
    for i,center in enumerate(centers):
        mask = p==i
        ul = center-meanDist
        br = center+meanDist
        xMask = np.logical_and(fullPoints[:,0]>=ul[0],fullPoints[:,0]<=br[0])
        yMask = np.logical_and(fullPoints[:,1]>=ul[1],fullPoints[:,1]<=br[1])
        coordMask = np.logical_and(xMask,yMask)
        mask = np.logical_and(mask,coordMask)
        masks.append(mask)
    return masks

def main():
    parser = argparse.ArgumentParser(description='Identify regions that could be refined more')
    parser.add_argument('--configFile', type=str, required=True)
    parser.add_argument('--errorFile', type=str)
    parser.add_argument('--partIndex', type=int)
    parser.add_argument('--outputFile', type=str)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--errorCutoff', type=float, default=0.05)
    args = parser.parse_args()
    with open(args.configFile) as file:
        config = yaml.load(file)

    with open(os.path.join(basePath,config['data_params']['cache_file']),'rb') as file:
        data = pickle.load(file)
    neutral = data['neutral']
    active = data['active']
    neutral = neutral[active].astype('float32')
    charts = data['vCharts']
    uv = data['uv'][:len(neutral)]
    mask = charts==args.partIndex
    with open(args.errorFile,'rb') as file:
        error = pickle.load(file)['vertex']

    cutoff = args.errorCutoff
    weight = error[mask]
    weight[error[mask]<cutoff] = 0
    newMasks = maskPoints(uv[mask],weight,k=args.k)
    newCharts = -np.ones(len(charts),dtype='int32')
    newUV = uv.copy()
    for i,newMask in enumerate(newMasks):
        newIndex = np.arange(len(newCharts))[mask][newMask]
        newCharts[newIndex] = i
        points = uv[mask][newMask]
        points = (points-np.min(points,0))/(np.max(points,0)-np.min(points,0))
        temp = newUV[mask]
        temp[newMask] = points
        newUV[mask] = temp

    if args.outputFile is not None:
        data['vCharts'] = newCharts
        data['uv'] = newUV
        with open(args.outputFile,'wb') as file:
            pickle.dump(data,file)

    plt.figure(0)
    for i in range(np.max(newCharts)+1):
        plt.plot(uv[newCharts==i,0],uv[newCharts==i,1],'.',markersize=4)
    plt.axis('equal')
    plt.title('Refinement components')
    plt.show()

if __name__=='__main__':
    main()

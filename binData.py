import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
import sys
import tqdm
import yaml

import PoseGenerator

def samplePoints(v,f,n):

    # Compute the area of each triangle in the base mesh
    vf = v[f.reshape(-1)].reshape((-1,3,3))
    a,b,c = vf[:,0],vf[:,1],vf[:,2]
    ab,ac = b-a,c-a
    cross = np.cross(ab,ac)
    area = np.sqrt(np.sum(np.square(cross),-1))/2
    
    # Sample from the triangles
    p = area/np.sum(area)
    face = np.random.choice(len(p),n)#,p=p)
    a = np.random.uniform(0,1,(n,2))
    idx = a[:,1]>a[:,0]
    a[idx,0] = a[idx,0]/2
    a[idx,1] = a[idx,1]-a[idx,0]
    idx = np.logical_not(idx)
    a[idx,1] = a[idx,1]/2
    a[idx,0] = a[idx,0]-a[idx,1]
    bary = np.stack([a[:,0],a[:,1],1-(a[:,0]+a[:,1])],axis=-1).astype('float32')

    return face,bary

def generateRandomSamples(controlFile,n):

    controlRange = []
    with open(controlFile,) as file:
        for line in file:
            parts = line.strip().split(' ')
            minVal = float(parts[3])
            maxVal = float(parts[4])
            controlRange.append([minVal,maxVal])
    controlRange = np.asarray(controlRange)

    samples = []
    for _ in range(n):
        sample = np.random.uniform(0,1,len(controlRange))
        sample = sample*(controlRange[:,1]-controlRange[:,0])+controlRange[:,0]
        samples.append(sample)

    res = dict(samples=samples)
    return res

def main():
    parser = argparse.ArgumentParser(description='Separate samples by similiarity')
    parser.add_argument('--configFile', type=str, required=True)
    parser.add_argument('--sampleFile', type=str)
    parser.add_argument('--cacheFile', type=str)
    parser.add_argument('--pointFile', type=str)
    parser.add_argument('--bins', type=int, default=10)
    parser.add_argument('--outputFile', type=str, required=True)
    args = parser.parse_args()
    with open(args.configFile) as file:
        config = yaml.load(file)
    with open(config['data_params']['cache_file'],'rb') as file:
        data = pickle.load(file)
    with open(config['data_params']['cache_file'],'rb') as file:
        uvData = pickle.load(file)
    if args.sampleFile is not None:
        with open(args.sampleFile,'rb') as file:
            sampleData = pickle.load(file)
    else:
        sampleData = generateRandomSamples(config['data_params']['control_file'],n=10000)
    neutral = data['neutral']
    active = data['active']
    neutral = neutral[active].astype('float32')
    neutralMean = np.mean(neutral,0)
    faces = data['faces']

    samples = sampleData['samples']
    if args.cacheFile is not None and os.path.isfile(args.cacheFile):
        with open(args.cacheFile,'rb' ) as file:
            cache = pickle.load(file)
        f,b = cache['f'],cache['b']
        cachePoints = cache['points']
        count = len(cachePoints)
        n = cachePoints.shape[1]
        points = np.zeros((len(samples),n,3),dtype='float32')
        points[:count] = cachePoints
    else:
        if args.pointFile:
            with open(args.pointFile,'rb') as file:
                data=pickle.load(file)
            f,b = data['faces'],data['bary']
            n = len(f)
        else: # Select points at random on the mesh if none are provided
            n = 10
            f,b = samplePoints(neutral,triMasked,n)
        points = np.zeros((len(samples),n,3),dtype='float32')
        count = 0

    # Gather the data
    pg = PoseGenerator.PoseGeneratorRemote(config['data_params']['control_file'],config['data_params']['geo_file'],'localhost',9001)
    pg.connect()
    try:
        pg.setActiveVertices(active)
        print('Generating sample meshes')
        for i in tqdm.trange(count,len(samples)):
            pg.setPose(samples[i])
            m = pg.getVertices()[active]
            faceMesh = m[faces.reshape(-1)].reshape((-1,3,3))
            p = np.sum(faceMesh[f]*b[...,np.newaxis],1)
            points[i] = p
            i = i + 1 # Needed for properly saving the cache
        pg.setPose(samples[5])
        mTest = pg.getVertices()[active]
    finally:
        if args.cacheFile is not None and count < len(samples):
            cache = dict(f=f,b=b,points=points[:i])
            with open(args.cacheFile,'wb') as file:
                pickle.dump(cache,file)
        pg.close()

    # Run PCA on the data
    x = points.reshape((len(samples),-1))
    x = x - np.mean(x,0)
    x = x / np.std(x,0)
    pca = PCA(n_components=1)
    res = pca.fit_transform(x)
    res = res.reshape(-1)

    hist,edges = np.histogram(res,bins=args.bins)
    bins = np.digitize(res,edges[1:-1])

    sampleData['faces'] = f
    sampleData['bary' ] = b
    sampleData['points'] = points
    sampleData['pca'] = res
    sampleData['bins'] = bins
    with open(args.outputFile,'wb') as file:
        pickle.dump(sampleData,file)

if __name__=='__main__':
   main()

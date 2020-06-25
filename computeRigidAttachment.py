import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from skimage.transform import resize
import sys
import trimesh
import tqdm
import yaml
from pathlib import Path

basePath = (Path(__file__).parent).resolve()

import PoseGenerator
import SSD

def modifyCaches(cacheFile, nonRigidParts):
    with open(cacheFile, 'rb') as f:
        cacheData = pickle.load(f)
    cacheData['rigidCharts'] = cacheData['vCharts'].copy()
    rigidParts = [x for x in np.unique(cacheData['vCharts']) if x not in nonRigidParts]
    print("Non-Rigid Parts: ", nonRigidParts)
    print("Rigid Parts: ", rigidParts)
    for part in rigidParts:
        cacheData['vCharts'][cacheData['vCharts'] == part] = -1
    for part in nonRigidParts:
        cacheData['rigidCharts'][cacheData['rigidCharts'] == part] = -1
    with open(cacheFile, 'wb') as f:
        pickle.dump(cacheData, f)

    print('vCharts: ', dict(zip(*np.unique(cacheData['vCharts'], return_counts=True))))
    print('rigidCharts: ', dict(zip(*np.unique(cacheData['rigidCharts'], return_counts=True))))

    return rigidParts

def rankVertices(usedIndex,rigidIndex,meshes,neutral):

    # Compute the transformation of the rigid component
    rigidNeutral = neutral[rigidIndex]
    rigidMesh = meshes[:,rigidIndex]
    transformations = [SSD.rigidRegister(rigidNeutral,m) for m in rigidMesh]

    # Compute the mesh positions given the transofrmations
    usedNeutral = neutral[usedIndex]
    usedMesh = meshes[:,usedIndex]
    deformed = np.asarray([usedNeutral.dot(R)+t for R,t in transformations]).astype('float32')
    diff = np.sqrt(np.sum(np.square(usedMesh-deformed),-1))
    error = np.mean(diff,0)

    return error

def computeRigidComponents(charts,meshes,neutral):

    rigid = []
    nonrigid = []

    for i in range(np.max(charts)+1):
        mask = charts==i
        neutralPart = neutral[mask]
        meshesPart = meshes[:,mask]
        transformations = [SSD.rigidRegister(neutralPart,m) for m in meshesPart]
        deformed = np.asarray([neutralPart.dot(R)+t for R,t in transformations])
        diff = np.sum(np.square(meshesPart-deformed),-1)
        error = np.mean(diff)
        if error < 1e-6:
            rigid.append(i)
        else:
            nonrigid.append(i)

    return rigid,nonrigid

def main():
    parser = argparse.ArgumentParser(description='Train deformation approximation with CNN')
    parser.add_argument('--configFile', type=str, required=True)
    parser.add_argument('--outputFile', type=str, required=True)
    parser.add_argument('--nonRigidParts', nargs='+', type=int)
    args = parser.parse_args()
    with open(args.configFile) as file:
        config = yaml.load(file)

    with open(os.path.join(basePath,config['data_params']['cache_file']),'rb') as file:
        data = pickle.load(file)
    neutral = data['neutral']
    active = data['active']
    charts = data['vCharts']
    neutral = neutral[active]

    if 'rigidCharts' in data:
        print('Using rigid data from cache file')
        nonrigid = list(set([i for i in charts if i != -1]))
        rigidPartsIdx = list(set([i for i in data['rigidCharts'] if i != -1]))
        hasRigid = True
    else:
        hasRigid = False

    # Create the data pipeline
    print('Loading control file '+config['data_params']['control_file'])
    pg = PoseGenerator.PoseGeneratorRemote(os.path.join(basePath,config['data_params']['control_file']),os.path.join(basePath,config['data_params']['geo_file']),'localhost',9001)
    pg.connect()
    pg.setActiveVertices(active)

    # Generate some training data
    n = 250
    meshes = np.zeros((n,len(neutral),3),dtype='float32')
    try:
        for i in tqdm.trange(n):
            pg.setRandomPose()
            m = pg.getVertices()[active]
            meshes[i] = m
    finally:
        pg.close()

    # modify cache with information on which parts are rigid
    if not hasRigid:
        if args.nonRigidParts is None:
            print('Computing rigid components')
            rigid,nonrigid = computeRigidComponents(charts,meshes,neutral)
        else:
            nonrigid = args.nonRigidParts
        rigidPartsIdx = modifyCaches(os.path.join(basePath,config['data_params']['cache_file']), nonrigid)

    with open(os.path.join(basePath,config['data_params']['cache_file']),'rb') as file:
        data = pickle.load(file)
    neutral = data['neutral']
    active = data['active']
    neutral = neutral[active].astype('float32')
    neutralMean = np.mean(neutral,0)
    faces = data['faces']
    charts = data['vCharts']
    if 'sample_file' in config['data_params']:
        sampleFile = os.path.join(basePath,config['data_params']['sample_file'])
    else:
        sampleFile = None

    # Identify the used vertices
    usedV = np.arange(len(neutral))[charts>-1]

    # Identify the rigid vertices
    rigidCharts = data['rigidCharts']
    rigidMax = np.max(rigidCharts)+1
    rigidParts = []
    for i in range(rigidMax):
        if np.sum(rigidCharts==i):
            rigidParts.append(np.arange(len(neutral))[rigidCharts==i])
    print('Computing attachment for ' + str(len(rigidParts)) + ' parts')

    # Rank the vertices
    errors = [rankVertices(usedV,rp,meshes,neutral) for rp in rigidParts]
    for i in range(len(errors)):
        print('part '+str(rigidPartsIdx[i]))
        print('\tmin(errors): '+str(np.min(errors[i])))
        print('\tmax(errors): '+str(np.max(errors[i])))
        print('\terrors<0.1: '+str(np.sum(errors[i]<0.1)))

    # Save the results
    output = dict(parts=rigidParts,rank=errors)
    with open(args.outputFile,'wb') as file:
        pickle.dump(output,file)
    
if __name__=='__main__':
    main()

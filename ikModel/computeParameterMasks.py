import argparse
import numpy as np
import os
import pickle
import sys
import tqdm
import yaml
from pathlib import Path

basePath = (Path(__file__).parent / '..').resolve()
sys.path.append(os.path.join(basePath, '.'))

import PoseGenerator

# Identify which rig paraemters affect the specified region of the mesh
def maskParameters(pg,faces):
    mask = np.zeros((len(faces),len(pg.createRandomPose())),dtype=bool)
    iterations = 2
    eps = 1e-4
    for itr in range(iterations):
        pose = np.asarray(pg.createRandomPose())
        pose = np.asarray(pg.defaultPose)
        pg.setPose(pose)
        mesh = pg.getVertices()[pg.active]
        newPose =  np.asarray(pg.createRandomPose())
        for i in tqdm.trange(len(pose)):
            pp = pose.copy()
            pp[i] = pg.poseRange[i][itr]
            pg.setPose(pp)
            mp = pg.getVertices()[pg.active]
            diff = np.abs(mp-mesh)
            for j in range(len(mask)):
                if np.max(diff[faces[j]]) > eps:
                    mask[j,i] = True
    return mask

def main():
    parser = argparse.ArgumentParser(description='Compute parameter masks for IK models')
    parser.add_argument('--configFile', type=str, required=True)
    parser.add_argument('--pointFile', type=str, required=True)
    parser.add_argument('--outputFile', type=str, required=True)
    parser.add_argument('--controlFile', type=str)
    args = parser.parse_args()
    with open(args.configFile) as file:
        config = yaml.load(file)

    with open(os.path.join(basePath,config['training_params']['approximation_config'])) as file:
        approxConfig = yaml.load(file)
    with open(os.path.join(basePath,approxConfig['data_params']['cache_file']),'rb') as file:
        data = pickle.load(file)
    with open(args.pointFile,'rb') as file:
        pointData = pickle.load(file)
    neutral = data['neutral']
    active = data['active']
    neutral = neutral[active].astype('float32')
    neutralMean = np.mean(neutral,0)
    faces = data['faces'][pointData['faces']]
    bary = pointData['bary']

    # Connect to the pose generator
    if args.controlFile is not None:
        controlFile = args.controlFile
    else:
        controlFile = os.path.join(basePath,config['data_params']['control_file'])
    pg = PoseGenerator.PoseGeneratorRemote(controlFile,os.path.join(basePath,approxConfig['data_params']['geo_file']),'localhost',9001)
    pg.connect()
    pg.setActiveVertices(active)

    # Find the controls that affect each chart
    mask = maskParameters(pg,faces)
    pg.close()
    maskInt = mask.astype('int32')
    pointData['parameter_mask'] = mask
    with open(args.outputFile,'wb') as file:
        pickle.dump(pointData,file)

if __name__=='__main__':
    main()

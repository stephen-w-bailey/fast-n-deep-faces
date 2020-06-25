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
def maskParameters(pg,charts):
    mask = np.zeros((np.max(charts)+1,len(pg.createRandomPose())),dtype=bool)
    vMask = np.zeros((len(charts),len(pg.createRandomPose())),dtype=bool)
    iterations = 2
    eps = 1e-4
    print('Generating mesh data')
    for itr in range(iterations):
        #pose = np.asarray(pg.createRandomPose())
        pose = np.asarray(pg.defaultPose)
        pg.setPose(pose)
        mesh = pg.getVertices()[pg.active]
        newPose =  np.asarray(pg.createRandomPose())
        for i in tqdm.trange(len(pose)):
            pp = pose.copy()
            #pp[i] = newPose[i]
            pp[i] = pg.poseRange[i][itr]
            pg.setPose(pp)
            mp = pg.getVertices()[pg.active]
            diff = np.abs(mp-mesh)
            vMask[np.sum(diff,-1)>eps,i] = True
            for j in range(len(mask)):
                if np.sum(charts==j) == 0:
                    continue
                if np.max(diff[charts==j]) > eps:
                    mask[j,i] = True
    return mask,vMask

def main():
    parser = argparse.ArgumentParser(description='Compute parameter masks for CNN models')
    parser.add_argument('--configFile', type=str, required=True)
    parser.add_argument('--outputFile', type=str, required=True)
    parser.add_argument('--controlFile', type=str)
    args = parser.parse_args()

    with open(args.configFile) as file:
        config = yaml.load(file)

    with open(os.path.join(basePath,config['data_params']['cache_file']),'rb') as file:
        data = pickle.load(file)
    neutral = data['neutral']
    active = data['active']
    neutral = neutral[active].astype('float32')
    neutralMean = np.mean(neutral,0)
    faces = data['faces']

    # Connect to the pose generator
    if args.controlFile is not None:
        controlFile = args.controlFile
    else:
        controlFile = config['data_params']['control_file']
    pg = PoseGenerator.PoseGeneratorRemote(os.path.join(basePath,controlFile),os.path.join(basePath,config['data_params']['geo_file']),'localhost',9001)
    pg.connect()
    pg.setActiveVertices(active)

    # Load the charts
    print('Loading chart file '+str(config['data_params']['cache_file']))
    with open(os.path.join(basePath,config['data_params']['cache_file']),'rb') as file:
        data = pickle.load(file)
    faces = data['originalFaces']
    numV = np.max(faces)+1
    charts = data['vCharts'][:numV]

    # Find the controls that affect each chart
    mask,vMask = maskParameters(pg,charts)
    pg.close()
    maskInt = mask.astype('int32')
    data['parameter_mask'] = mask
    data['parameter_v_mask'] = vMask
    with open(args.outputFile,'wb') as file:
        pickle.dump(data,file)
    #    for m in maskInt:
    #        file.write(','.join([str(i) for i in m])+'\n')

if __name__=='__main__':
    main()

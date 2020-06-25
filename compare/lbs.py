import argparse
import numpy as np
import os
import pickle
import scipy.io as sio
import sys
import tqdm
from pathlib import Path

basePath = (Path(__file__).parent / '..').resolve()
sys.path.append(os.path.join(basePath, '.'))

import errors
import PoseGenerator
import SSD

def main():
    parser = argparse.ArgumentParser(description='Use SSDRB to generate LBS weights')
    parser.add_argument('--cacheFile', type=str, required=True)
    parser.add_argument('--controlFile', type=str, required=True)
    parser.add_argument('--geoFile', type=str, required=True)
    parser.add_argument('--animFile', type=str)
    parser.add_argument('--sampleFile', type=str)
    parser.add_argument('--outputMesh', type=str)
    parser.add_argument('--sampleCount', type=int, default=1000)
    parser.add_argument('--bones', type=int, default=12)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--ssdFile', type=str)
    args = parser.parse_args()

    with open(args.cacheFile,'rb') as file:
        data = pickle.load(file)

    if args.sampleFile is not None and os.path.isfile(args.sampleFile):
        with open(args.sampleFile,'rb') as file:
            samples = pickle.load(file)
    else:
        samples = None

    active = data['active']
    default = data['neutral'][active]
    faces = data['faces']
    pg=PoseGenerator.PoseGeneratorRemote(args.controlFile,args.geoFile,'localhost',9001)
    pg.connect()
    mesh = []
    sampleMesh = []

    sampleCount = args.sampleCount * 2
    if args.animFile is not None and os.path.isfile(args.animFile):
        with open(args.animFile,'rb') as file:
            anim = pickle.load(file)
    else:
        np.random.seed(9001)
        anim = [pg.createRandomPose() for _ in range(sampleCount)]

    try:
        pg.setActiveVertices(active)
        print('Generating meshes')
        order = range(len(anim))
        for index in tqdm.tqdm(order):
            pose = anim[index]
            pg.setPose(pose)
            v = pg.getVertices()[active]
            mesh.append(v)
        if samples is not None:
            print('Generating samples')
            for pose in tqdm.tqdm(samples):
                pg.setPose(pose)
                v = pg.getVertices()[active]
                sampleMesh.append(v)
        else:
            sampleAnim = anim[sampleCount//2:]
            sampleMesh = mesh[sampleCount//2:]
            anim = anim[:sampleCount//2]
            mesh = mesh[:sampleCount//2]

    finally:
        pg.close()

    if args.sampleFile is not None and not os.path.isfile(args.sampleFile):
        with open(args.sampleFile,'wb') as file:
            pickle.dump(sampleAnim,file)
    if args.animFile is not None and not os.path.isfile(args.animFile):
        with open(args.animFile,'wb') as file:
            pickle.dump(anim,file)

    mesh = np.asarray(mesh)
    sampleMesh = np.asarray(sampleMesh)

    # Train model
    ssd = SSD.SSD()
    if args.ssdFile is None or not os.path.isfile(args.ssdFile):
        ssd.runSSD(default,sampleMesh,args.bones,args.k,faces=faces)
        if args.ssdFile is not None:
            with open(args.ssdFile,'wb') as file:
                out = dict(ssdWeights=ssd.weights,
                           ssdRestBones=ssd.restBones,
                           ssdRest=ssd.rest)
                pickle.dump(out,file)
    else:
        with open(args.ssdFile,'rb') as file:
            params = pickle.load(file)
        ssd.weights = params['ssdWeights']
        ssd.restBones = params['ssdRestBones']
        ssd.rest = params['ssdRest']

    # Evaluate model
    bones = np.repeat(ssd.restBones[np.newaxis],len(mesh),axis=0)
    error = ssd.getFitError(mesh,bones)
    prevError = error
    print('Initial error: '+str(error))
    for _ in range(20):
        bones = ssd.computeBones(mesh,bones)
        error = ssd.getFitError(mesh,bones)
        print('Error: '+str(error))
        if prevError-error < 1e-4:
            break
        else:
            prevError = error
    print('Final error: '+str(error))

    lbsMesh = []
    es = []
    normalErrors = []
    adjList = errors.buildFaceAdjList(faces)
    for i in tqdm.trange(len(bones)):
        v = ssd.computeMesh(bones[i])
        source = mesh[i]
        lbsMesh.append(ssd.computeMesh(bones[i]))
        ne = errors.normalError(source,v,faces)
        normalErrors.append(ne)
    diff = np.mean(np.sqrt(np.sum(np.square(lbsMesh-mesh),-1)),-1)
    print('Computed error: '+str(np.mean(diff)))
    print('Normal error: '+str(np.mean(normalErrors)))

    if args.outputMesh is not None:
        res = dict(neutral=data['neutral'],
                   active = data['active'])
        res['lbs'] = lbsMesh
        sio.savemat(args.outputMesh,res)

if __name__=='__main__':
    main()

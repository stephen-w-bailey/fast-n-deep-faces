import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from PySide2 import QtWidgets
from skimage.transform import resize
import scipy.io as sio
import sys
import tensorflow as tf
import trimesh
import tqdm
import yaml
from pathlib import Path
from collections import namedtuple

basePath = (Path(__file__).parent / '..').resolve()
sys.path.append(os.path.join(basePath, '.'))
sys.path.append(os.path.join(basePath, 'compare'))

import cnnModel
import cnnOpt
import dataLoader
import denseModel
from errors import normalError
import PoseGenerator
import render
import UVGenerator
import rigidDeformer

logging.getLogger().setLevel(logging.INFO)

def buildModel(config,pose,addNeutral=True):

    with open(config['data_params']['cache_file'],'rb') as file:
        data = pickle.load(file)
    parts = data['vCharts']
    faces = data['faces']
    neutral = data['neutral'][data['active']].astype('float32')
    uvs = data['uv']
    if 'parameter_mask' in data:
        mask = data['parameter_mask']
    else:
        mask = None

    # Create the model
    partCount = np.max(parts)+1
    data = {'pose':pose}
    usedVerts = []
    usedUVs = []
    for i in range(partCount):
        if np.sum(parts==i) > 0:
            data['image-'+str(i)] = tf.ones(1)
        else:
            data['image-'+str(i)] = None
        ref = faces.reshape(-1)
        idx = np.arange(len(neutral))[parts==i]
        if len(idx) == 0:
            continue
        usedFaces = [True if v in idx else False for v in ref]
        usedFaces = np.sum(np.asarray(usedFaces).reshape((-1,3)),-1) > 0
        faceIdx = np.arange(len(faces))[usedFaces]
        idx = np.arange(len(parts))[parts==i]
        uv = uvs[idx]
        usedUVs.append(uv)
        usedVerts.append(idx)
    idx = np.concatenate(usedVerts)
    linear = np.zeros(neutral.shape,dtype='float32')
    if addNeutral:
        linear[idx] = neutral[idx]
    else:
        neutral = linear
    data['linear'] = linear
    dataset = namedtuple('Dataset','mask usedUVs usedVerts vCharts')(mask,usedUVs,usedVerts,parts)
    model = cnnModel.buildModel(data,dataset,neutral,config)

    return model

def buildRigid(config,mesh):
    with open(config['data_params']['cache_file'],'rb') as file:
        data = pickle.load(file)
    parts = data['vCharts']
    neutral = data['neutral'][data['active']].astype('float32')
    mask = np.arange(len(parts))[parts>-1]
    deformer = rigidDeformer.RigidDeformer(neutral,config['data_params']['rigid_files'],mask)
    mesh = deformer.deformTF(mesh[0])[np.newaxis]
    return mesh

def main():
    parser = argparse.ArgumentParser(description='Evaluate deformation approximation with CNN')
    parser.add_argument('--configFile', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--animFile', type=str, required=True)
    parser.add_argument('--outputMeshes', type=str)
    args = parser.parse_args()
    with open(args.configFile) as file:
        config = yaml.load(file)

    with open(os.path.join(basePath,config['data_params']['cache_file']),'rb') as file:
        data = pickle.load(file)
    neutral = data['neutral']
    active = data['active']
    parts = data['vCharts']
    neutral = neutral[active].astype('float32')
    neutralMean = np.mean(neutral,0)
    faces = data['faces']
    cacheData = data

    # Create the data pipeline
    pg = PoseGenerator.PoseGeneratorRemote(os.path.join(basePath,config['data_params']['control_file']),os.path.join(basePath,config['data_params']['geo_file']),'localhost',9001)
    pg.connect()
    pg.setActiveVertices(active)

    # Load the test data or generate it if the file doesn't exist
    if args.animFile is not None and os.path.isfile(args.animFile):
        with open(args.animFile,'rb') as file:
            anim = pickle.load(file)
    else:
        np.random.seed(9001)
        sampleCount = 100
        anim = [pg.createRandomPose() for _ in range(sampleCount)]
        anim = np.asarray(anim)
        with open(args.animFile,'wb') as file:
            pickle.dump(anim,file)
            
    if 'linear_file' not in config['data_params']:
        config['data_params']['linear_file'] = None
    else:
        config['data_params']['linear_file'] = os.path.join(basePath,config['data_params']['linear_file'])
    dataset = dataLoader.AnimationLoader(pg,anim,os.path.join(basePath,config['data_params']['cache_file']),linearModel=config['data_params']['linear_file'])
    if 'channels' in config['model_params']:
        imageSize = 2**len(config['model_params']['channels'])
        if 'initial_image' in config['model_params']:
            imageSize *= config['model_params']['initial_image'][0]
    else:
        imageSize = 64
    batchSize = 1
    data = dataset.createDataset(batchSize,imageSize)

    # Build the model
    if 'base_config' in config['model_params']:
        with open(config['model_params']['base_config']) as file:
            baseConfig = yaml.load(file)
        baseModel = buildModel(baseConfig,data['pose'])
        with tf.variable_scope('refine'):
            model = buildModel(config,data['pose'],addNeutral=False)
        model['output'] = model['output'] + baseModel['output']
        if 'rigid_files' in config['data_params']:
            model['output'] = buildRigid(baseConfig,model['output'])
            baseModel['output'] = buildRigid(baseConfig,baseModel['output'])
        elif 'rigid_files' in baseConfig['data_params']:
            model['output'] = buildRigid(baseConfig,model['output'])
            baseModel['output'] = buildRigid(baseConfig,baseModel['output'])
    else:
        model = buildModel(config,data['pose'])
        if 'rigid_files' in config['data_params']:
            model['output'] = buildRigid(config,model['output'])
        baseModel = None

    save = tf.train.Saver()

    refineErrors = dict(l1=[],l2=[],normal=[])
    baseErrors = dict(l1=[],l2=[],normal=[])
    if args.outputMeshes:
        outputs = dict(gt=[],base=[])
        if baseModel is not None:
            outputs['refine'] = []
    else:
        outputs = None

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if os.path.isfile(args.checkpoint):
            checkpoint = args.checkpoint
        else:
            checkpoint = tf.train.latest_checkpoint(args.checkpoint)
        save.restore(sess,checkpoint)

        meshes = dict(gt=data['mesh'])
        if baseModel is not None:
            meshes['refine'] = model['output']
            meshes['base'] = baseModel['output']
        else:
            meshes['base'] = model['output']

        def addErrors(mesh,gt,errors):
            diff = np.sum(np.abs(mesh-gt),-1)
            error = np.mean(diff)
            errors['l1'].append(error)
            diff = np.sqrt(np.sum(np.square(mesh-gt),-1))
            error = np.mean(diff)
            errors['l2'].append(error)
            errors['normal'].append(normalError(mesh[0],gt[0],faces))

        for _ in tqdm.trange(len(anim)):
            res = sess.run(meshes)
            gt = res['gt']
            if 'refine' in res:
                addErrors(res['refine'],gt,refineErrors)
            addErrors(res['base'],gt,baseErrors)

            if outputs is not None:
                for k in outputs:
                    outputs[k].append(res[k][0])

        pg.close()

    if args.outputMeshes:
        outputs['neutral'] = cacheData['neutral']
        outputs['active'] = cacheData['active']
        print('Saving '+str(outputs.keys())+' to file '+args.outputMeshes)
        sio.savemat(args.outputMeshes,outputs)

    if len(refineErrors['l1']) > 0:
        print('Refinement Error:')
        for k in refineErrors:
            print(k+': '+str(np.mean(refineErrors[k])))
    print('Base Error:')
    for k in baseErrors:
        print(k+': '+str(np.mean(baseErrors[k])))

if __name__=='__main__':
    main()

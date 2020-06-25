import argparse
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from PySide2 import QtWidgets
from skimage.transform import resize
import sys
import tensorflow as tf
from tensorflow.python.client import timeline
import time
import trimesh
import tqdm
import yaml
from pathlib import Path

basePath = (Path(__file__).parent / '..').resolve()
sys.path.append(os.path.join(basePath, '.'))

import cnnModel
import cnnOpt
import dataLoader
import denseModel
import render
import rigidDeformer
import UVGenerator

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
        usedFaces = np.sum(np.asarray(usedFaces).reshape((-1,3)),-1) == 3
        faceIdx = np.arange(len(faces))[usedFaces]
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
    dataset = namedtuple('Dataset','mask usedUVs usedVerts')(mask,usedUVs,usedVerts)
    model = cnnModel.buildModel(data,dataset,neutral,config)

    return model

def buildRigid(config,mesh):
    print('Applying the rigid deformer')
    with open(config['data_params']['cache_file'],'rb') as file:
        data = pickle.load(file)
    parts = data['vCharts']
    neutral = data['neutral'][data['active']].astype('float32')
    mask = np.arange(len(parts))[parts>-1]
    deformer = rigidDeformer.RigidDeformer(neutral,config['data_params']['rigid_files'],mask)
    mesh = deformer.deformTF(mesh[0])[np.newaxis]
    return mesh

def evaluateCharts(vCharts,approx,groundTruth,linear):
    chartErrors = {}
    linearErrors = {}
    for i in range(np.max(vCharts)+1):
        if np.sum(vCharts==i) == 0:
            continue
        else:
            chartErrors[i] = []
            linearErrors[i] = []
    
    for m,gt,lin in zip(approx,groundTruth,linear):
        diff = np.sqrt(np.sum(np.square(m-gt),-1))
        diffLinear = np.sqrt(np.sum(np.square(lin-gt),-1))
        for key in chartErrors:
            diffChart = diff[vCharts==key]
            diffLin = diffLinear[vCharts==key]
            chartErrors[key].append(diffChart)
            linearErrors[key].append(diffLin)

    for key in chartErrors:
        error = np.stack(chartErrors[key],axis=0)
        meanError = np.mean(error)
        print('Chart '+str(key)+' L2 loss: '+str(meanError)+' ('+str(len(error))+')')
        error = np.stack(linearErrors[key],axis=0)
        meanError = np.mean(error)
        print('Linear '+str(key)+' L2 loss: '+str(meanError)+' ('+str(len(error))+')')

def main():
    parser = argparse.ArgumentParser(description='Evaluate deformation approximation with CNN')
    parser.add_argument('--configFile', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sampleFile', type=str, required=True)
    parser.add_argument('--n', type=int, default=10)
    args = parser.parse_args()
    with open(args.configFile) as file:
        config = yaml.load(file)

    with open(config['data_params']['cache_file'],'rb') as file:
        data = pickle.load(file)
    neutral = data['neutral']
    active = data['active']
    neutral = neutral[active].astype('float32')
    neutralMean = np.mean(neutral,0)
    faces = data['faces']

    # Load sample poses
    with open(args.sampleFile,'rb') as file:
        samples = pickle.load(file)
    posePH = tf.placeholder(tf.float32,samples.shape[1])
    pose = posePH[np.newaxis]
    samples = samples[np.random.choice(len(samples),args.n)]

    # Build the model
    if 'base_config' in config['model_params']:
        with open(config['model_params']['base_config']) as file:
            baseConfig = yaml.load(file)
        baseModel = buildModel(baseConfig,pose)
        with tf.variable_scope('refine'):
            model = buildModel(config,pose,addNeutral=False)
        model['output'] = model['output'] + baseModel['output']
        print('using refinement model')

        with tf.device('/CPU:0'):
            if 'rigid_files' in config['data_params']:
                model['output'] = buildRigid(config,model['output'])
            elif 'rigid_files' in baseConfig['data_params']:
                model['output'] = buildRigid(baseConfig,model['output'])
            else:
                print('Not applying the rigid deformer')
            if 'rigid_files' in baseConfig['data_params']:
                baseModel['output'] = buildRigid(baseConfig,baseModel['output'])

    else:
        model = buildModel(config,pose)
        baseModel = None

        with tf.device('/CPU:0'):
            if 'rigid_files' in config['data_params']:
                model['output'] = buildRigid(config,model['output'])

    save = tf.train.Saver()

    print('Timing results')
    times = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if os.path.isfile(args.checkpoint):
            checkpoint = args.checkpoint
        else:
            checkpoint = tf.train.latest_checkpoint(args.checkpoint)
        save.restore(sess,checkpoint)

        feed_dict = {posePH:samples[0]}
        mesh = sess.run(model['output'],feed_dict=feed_dict)

        for i,s in enumerate(samples):
            feed_dict = {posePH:s}
            start = time.time()
            mesh = sess.run(model['output'],feed_dict=feed_dict)
            end = time.time()
            if i < 10:
                print('elapsed: {:.2f}ms'.format((end-start)*1000))
            times.append(end-start)

    print('Average time: {:.2f}ms'.format(np.mean(times)*1000))

if __name__=='__main__':
    main()

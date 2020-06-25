import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from PySide2 import QtWidgets
from skimage.transform import resize
import sys
import tensorflow as tf
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
import PoseGenerator
import render
import UVGenerator

def main():
    parser = argparse.ArgumentParser(description='Train deformation approximation with CNN')
    parser.add_argument('--configFile', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
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
    charts = data['vCharts']

    # Create the data pipeline
    pg = PoseGenerator.PoseGeneratorRemote(os.path.join(basePath,config['data_params']['control_file']),os.path.join(basePath,config['data_params']['geo_file']),'localhost',9001)
    pg.connect()
    pg.setActiveVertices(active)
    if 'linear_file' not in config['data_params']:
        config['data_params']['linear_file'] = None
    else:
        config['data_params']['linear_file'] = os.path.join(basePath,config['data_params']['linear_file'])

    with tf.variable_scope('data_loader'):
        dataset = dataLoader.ImageDataLoader(pg,os.path.join(basePath,config['data_params']['cache_file']),linearModel=config['data_params']['linear_file'])
        if 'channels' in config['model_params']:
            imageSize = 2**len(config['model_params']['channels'])
            if 'initial_image' in config['model_params']:
                imageSize *= config['model_params']['initial_image'][0]
        else:
            imageSize = 64
        batchSize = config['training_params']['batch_size']
        data = dataset.createDataset(batchSize,imageSize)
        data['faces'] = faces

    # Build the model
    base = np.zeros(neutral.shape,dtype='float32') # Fill missing points with zeros
    modelType = 'cnn'
    if 'model_type' in config['model_params']:
        modelType = config['model_params']['model_type']
    print('Building '+str(modelType)+' model')
    if modelType == 'cnn':
        model = cnnModel.buildModel(data,dataset,base,config)
    elif modelType == 'dense':
        model = denseModel.buildModel(data,dataset,base,config)
    else:
        raise ValueError('Unrecognized model type '+str(modelType))
    usedVerts = np.concatenate(dataset.usedVerts,axis=0)
    loss = cnnOpt.buildLoss(data,model,config,usedVerts=usedVerts)
    optOps = cnnOpt.buildOpt(loss,config)
    cnnOpt.buildSummaries(data,model,loss,dataset.usedUVs)

    lrs = [float(lr) for lr in config['training_params']['lr']]
    steps = [int(s) for s in config['training_params']['steps']]
    with tf.train.MonitoredTrainingSession(checkpoint_dir=args.checkpoint,
                                           save_checkpoint_steps=50000,
                                           save_summaries_steps=250,
                                           save_summaries_secs=None) as sess:
        try:
            print('Beginning model training')
            for lr,step in zip(lrs,steps):
                print('Setting learning rate to '+str(lr))
                sess.run(optOps['lrAssign'],feed_dict={optOps['lrPH']:lr})
                for _ in tqdm.trange(step):
                    sess.run(optOps['opt'])
        except KeyboardInterrupt:
            print('Stopping early due to keyboard interrupt')

    pg.close()
    
if __name__=='__main__':
    main()

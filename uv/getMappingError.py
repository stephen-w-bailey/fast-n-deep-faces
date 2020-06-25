import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from skimage.transform import resize
import sys
import tensorflow as tf
import time
import tqdm
import trimesh
import yaml
from pathlib import Path

basePath = (Path(__file__).parent / '..').resolve()

sys.path.append(os.path.join(basePath,'cnnModel'))
sys.path.append(os.path.join(basePath, '.'))

import cnnOpt
import dataLoader
import PoseGenerator
import UVGenerator

def main():
    parser = argparse.ArgumentParser(description='Visualize errors from image mapping')
    parser.add_argument('--configFile', type=str, required=True)
    parser.add_argument('--outputError', type=str)
    parser.add_argument('--partIndex', type=int, required=True)
    parser.add_argument('--n', type=int, default=250)
    args = parser.parse_args()

    with open(args.configFile) as file:
        config = yaml.load(file)

    with open(os.path.join(basePath,config['data_params']['cache_file']),'rb') as file:
        data = pickle.load(file)
    neutral = data['neutral']
    active = data['active']
    neutral = neutral[active].astype('float32')
    meshMean = np.mean(neutral,0)
    neutral = neutral - meshMean
    faces = data['faces']
    charts = data['vCharts']
    uv = data['uv'][:len(neutral)]

    if 'sample_file' in config['data_params']:
        samples = os.path.join(basePath,config['data_params']['sample_file'])
    else:
        samples = None

    print('charts shape: '+str(charts.shape))
    print('neutral shape: '+str(neutral.shape))

    pg = PoseGenerator.PoseGeneratorRemote(os.path.join(basePath,config['data_params']['control_file']),os.path.join(config['data_params']['geo_file']),'localhost',9001)
    pg.connect()
    
    pg.setActiveVertices(active)
    if 'linear_file' not in config['data_params']:
        config['data_params']['linear_file'] = None
    else:
        config['data_params']['linear_file'] = os.path.join(basePath,config['data_params']['linear_file'])

    dataset = dataLoader.ImageDataLoader(pg,os.path.join(basePath,config['data_params']['cache_file']),linearModel=config['data_params']['linear_file'],makeImages=True)
    if 'channels' in config['model_params']:
        imageSize = 2**len(config['model_params']['channels'])
        if 'initial_image' in config['model_params']:
            imageSize = imageSize * config['model_params']['initial_image'][0]
    else:
        imageSize = 64
    batchSize = 1
    data = dataset.createDataset(batchSize,imageSize)
    total = 0
    while 'image-'+str(total) in data:
        total += 1
    images = [data['image-'+str(i)] for i in range(total) if data['image-'+str(i)].dtype != tf.string]
    imgMesh = UVGenerator.mapImagesToMesh(images,dataset.usedUVs,dataset.usedVerts,data['mesh']-data['linear'])
    imgMesh += data['linear']

    def fn(v):
        n = cnnOpt.faceNormals(v,faces)
        vn = cnnOpt.vertexNormals(v,n,faces)
        return vn
    vnGT = tf.map_fn(fn,data['mesh'])
    vnApprox = tf.map_fn(fn,imgMesh)

    with tf.Session() as sess:
        gt = []
        approx = []
        gtVN = []
        approxVN = []
        print('Generating data')
        for i in tqdm.trange(args.n):
            g,a,gn,an = sess.run([data['mesh'],imgMesh,vnGT,vnApprox])
            gt.append(g[0])
            approx.append(a[0])
            gtVN.append(gn[0])
            approxVN.append(an[0])

    # Compute the difference between the original mesh and the reconstructed mesh from the image
    gt = np.asarray(gt)
    approx = np.asarray(approx)
    approx[:,charts<0] = gt[:,charts<0]
    diff = np.sum(np.square(gt-approx),-1)
    diff = np.mean(diff,0)
    gtVN = np.asarray(gtVN)
    approxVN = np.asarray(approxVN)
    diffN = np.sum(np.square(gtVN-approxVN),-1)
    diffN = np.mean(diffN,0)
    diffN[np.isnan(diffN)] = 0

    if args.outputError is not None:
        data = dict(vertex=diff,normal=diffN)
        with open(args.outputError,'wb') as file:
            pickle.dump(data,file)

    time.sleep(5) # Avoid error whith dataloader caching after pg is closed
            
    pg.close()
    
if __name__=='__main__':
    main()

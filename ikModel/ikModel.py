import numpy as np
import pickle
import tensorflow as tf
import yaml

import ikOpt

def buildDense(x,layers,outDim):
    for i,layer in enumerate(layers):
        with tf.variable_scope('dense-'+str(i)):
            x = tf.layers.dense(x,layer,activation=tf.nn.leaky_relu)
    with tf.variable_scope('dense-out'):
        out = tf.layers.dense(x,outDim)
    return out

def combineParts(parts,mask,defaultPose):
    with tf.variable_scope('combiner'):
        maskSum = np.sum(mask,0)
        unusedPart = maskSum==0
        maskSum[maskSum==0] = 1
        parts = tf.stack(parts,axis=1)
        parts = tf.reduce_sum(parts*mask,1)/maskSum
        x = defaultPose*unusedPart + (1-unusedPart)*parts
        return x

def buildModel(data,config):
    
    with tf.variable_scope('ik-model'):
        if 'pose' in data:
            outDim = data['pose'].get_shape().as_list()[1]
        else:
            with open(config['data_params']['range_file']) as file:
                outDim = len([None for _ in file])
            
        x = data['points']

        axes = dict(x=0,y=1,z=2)
        index = []
        for c in config['model_params']['input_dims']:
            index.append(axes[c])
        x = tf.gather(x,index,axis=2)

        if 'normalize_points' in config['training_params'] and config['training_params']['normalize_points']:
            moments = tf.nn.moments(x,axes=[1])
            x = x-moments[0][:,np.newaxis]
            x = x/moments[1][:,np.newaxis]
        else:
            with open(config['training_params']['approximation_config']) as file:
                approxConfig = yaml.load(file)
            with open(approxConfig['data_params']['cache_file'],'rb') as file:
                approxData = pickle.load(file)
            neutral = approxData['neutral'][approxData['active']]
            mean = np.mean(neutral,0)
            std = np.std(neutral,0)
            axisMap = {'x':0,'y':1,'z':2}
            axes = []
            for c in config['model_params']['input_dims']:
                axes.append(axisMap[c])
            mean = mean[axes]
            std = std[axes]
            x = (x-mean)/std

        layers = config['model_params']['layers']
        if 'parts' in config['model_params']:
            with open(config['model_params']['mask'],'rb') as file:
                mask = pickle.load(file)['parameter_mask']
            modelParts = []
            maskParts = []
            for i,p in enumerate(config['model_params']['parts']):
                with tf.variable_scope('part-'+str(i)):
                    xPart = tf.gather(x,p,axis=1)
                    xPart = tf.layers.flatten(xPart)
                    modelParts.append(buildDense(xPart,layers,outDim))
                    maskParts.append(np.max(mask[p],0))
            maskParts = np.stack(maskParts,0)
            defaultPose = ikOpt.loadDefaultPose(config['data_params']['range_file'])[0]
            x = combineParts(modelParts,maskParts,defaultPose)
        else:
            x = tf.layers.flatten(x)
            x = buildDense(x,layers,outDim)

    model = dict(output=x)
    return model

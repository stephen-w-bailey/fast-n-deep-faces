import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append('..')

import denseModel
import UVGenerator

def upsample(tensor):
    shape = tensor.get_shape().as_list()
    x = tf.reshape(tensor,(-1,shape[1],1,shape[2],1,shape[3]))
    x = tf.tile(x,(1,1,2,1,2,1))
    x = tf.reshape(x,(-1,shape[1]*2,shape[2]*2,shape[3]))
    return x

def convLayer(tensor, filters, size, stride=(1,1), activation=None, norm=True, name='conv', reuse=tf.AUTO_REUSE):
    def _layer(tensor):
        with tf.variable_scope(name,reuse=reuse,use_resource=True):

            init = tf.truncated_normal_initializer(stddev=0.02)
            x = tf.layers.conv2d(tensor,filters,size,stride,padding='SAME',
                                 kernel_initializer=init,
                                 reuse=reuse,use_bias=not norm)
            tf.add_to_collection('checkpoints',x)
            if norm:
                x = instanceNorm(x,reuse=reuse)
            if activation is not None:
                x = activation(x)
            return x

    return _layer(tensor)

def instanceNorm(tensor,name='instanceNorm',reuse=tf.AUTO_REUSE):
    with tf.variable_scope(name,reuse=reuse,use_resource=True):
        shape = tensor.get_shape().as_list()
        mu,sigma = tf.nn.moments(tensor,[1,2],keep_dims=True)
        a = tf.get_variable('a',[shape[-1]],initializer=tf.truncated_normal_initializer(mean=1.0,stddev=0.02))
        b = tf.get_variable('b',initializer=tf.zeros([shape[-1]]))
        eps = 1e-3
        norm = (tensor-mu)/tf.sqrt(sigma+eps)
    return a*norm+b

def buildImageModel(pose,config):

    # Build a model that does upsample->conv->norm->relu->...
    if 'norm' in config['model_params']:
        norm = config['model_params']['norm']
    else:
        norm = False
    if 'activation' in config['model_params']:
        activations = dict(relu=tf.nn.relu,leaky=tf.nn.leaky_relu)
        activation = activations[config['model_params']['activation']]
    else:
        activation = tf.nn.relu
    channels = [int(c) for c in config['model_params']['channels']]
    if 'dense' in config['model_params']:
        x = pose
        for i,l in enumerate(config['model_params']['dense']):
            with tf.variable_scope('dense-'+str(i)):
                x = tf.layers.dense(x,l,activation)
        l = np.prod(config['model_params']['initial_image'])
        with tf.variable_scope('dense-final'):
            x = tf.layers.dense(x,l)
            x = tf.reshape(x,[-1]+list(config['model_params']['initial_image']))
    else:
        x = pose[:,np.newaxis,np.newaxis]
    for i,c in enumerate(channels):
        with tf.variable_scope('level-'+str(i)):
            x = upsample(x)
            x = convLayer(x,c,(3,3),activation=activation,norm=norm)
    if 'extra_layer' in config['model_params'] and config['model_params']['extra_layer']:
        with tf.variable_scope('level-extra'):
            x = convLayer(x,c,(3,3),activation=activation,norm=norm)
    with tf.variable_scope('level-'+str(len(channels))):
        x = convLayer(x,3,(1,1),activation=None,norm=False)
    return x

def buildModel(data,dataset,base,config):
    if 'model_type' in config['model_params'] and config['model_params']['model_type'] == 'dense':
        print('building dense model')
        return denseModel.buildModel(data,dataset,base,config)

    total = 0
    while 'image-'+str(total) in data:
        total += 1
    model = {}
    pose = data['pose']
    images = []
    for i in range(total):
        if data['image-'+str(i)] is None or (hasattr(data['image-'+str(i)],'dtype') and data['image-'+str(i)].dtype == tf.string):
            continue
        with tf.variable_scope('image-'+str(i)):
            if dataset.mask is not None:
                index = np.arange(dataset.mask.shape[1])[dataset.mask[i]]
                imageInput = tf.gather(pose,index,axis=1)
            else:
                imageInput = pose
            image = buildImageModel(imageInput,config)
            model['cnn'+str(i)] = image
            images.append(image)
    imgMesh = UVGenerator.mapImagesToMesh(images,dataset.usedUVs,dataset.usedVerts,base)
    model['images'] = images
    model['imageMesh'] = imgMesh
    if 'linear' in data:
        mesh = imgMesh + data['linear']
    else:
        mesh = imgMesh
    model['output'] = mesh
    return model

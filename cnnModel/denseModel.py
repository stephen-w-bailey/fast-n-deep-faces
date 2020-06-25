import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

def buildDenseModel(pose,outDim,config):

    if 'activation' in config['model_params']:
        activations = dict(relu=tf.nn.relu,leaky=tf.nn.leaky_relu)
        activation = activations[config['model_params']['activation']]
    else:
        activation = tf.nn.relu

    layers = [int(c) for c in config['model_params']['layers']]
    x = pose
    for i,c in enumerate(layers):
        with tf.variable_scope('layer-'+str(i)):
            x = tf.layers.dense(x,c,activation=activation)
    with tf.variable_scope('layer-'+str(len(layers))):
        x = tf.layers.dense(x,outDim)
    return x

def buildModelOld(data,dataset,base,config):
    vCharts = dataset.vCharts
    total = np.max(dataset.vCharts)+1
    model = {}
    pose = data['pose']
    parts = []
    charts = []
    mesh = data['linear']
    if isinstance(mesh,np.ndarray) and len(mesh.shape) == 2:
        mesh = mesh[np.newaxis]
    for i in range(total):
        if np.sum(vCharts==i) == 0:
            continue
        with tf.variable_scope('dense-'+str(i)):
            idx = np.arange(len(vCharts))[vCharts==i]
            part = tf.gather(mesh,idx,axis=1)
            dense = buildDenseModel(pose,len(idx)*3,config)
            part = tf.reshape(dense,(-1,len(idx),3))
            #part = np.zeros((1,len(idx),3),dtype='float32')
            #part = tf.constant(part,dtype=tf.float32)
            #part = tf.tile(part,(tf.shape(mesh)[0],1,1))
            if i == 15:
                part = tf.Print(part,[tf.reduce_mean(tf.square(part))],'Mean(part): ')
            model['dense'+str(i)] = part
            parts.append(part)
            charts.append(idx)
    usedVerts = np.concatenate(charts)
    unusedVerts = np.asarray([i for i in range(len(vCharts)) if i not in usedVerts])
    verts = np.concatenate((usedVerts,unusedVerts),0)

    #print('base type: '+str(base.shape))
    #print('mesh type: '+str(mesh.shape))
    #plt.plot(mesh[0,:,0],mesh[0,:,1],'.',markersize=1)
    #plt.plot(mesh[0,usedVerts,0],mesh[0,usedVerts,1],'.',markersize=4)
    #plt.axis('equal')
    #plt.show()
    #stuff

    if isinstance(base,np.ndarray):
        baseTF = tf.constant(base[unusedVerts],dtype=tf.float32)
        if len(base.shape) == 2:
            baseTF = tf.tile(baseTF[np.newaxis],(tf.shape(mesh)[0],1,1))
        base = baseTF
    else:
        base = tf.gather(base,unusedVerts,axis=-2)
    vs = tf.concat(parts+[base],axis=1)
    vOrder = np.argsort(verts)
    denseMesh = tf.gather(vs,vOrder,axis=1)
    model['denseMesh'] = denseMesh
    if 'linear' in data:
        mesh = denseMesh + data['linear']
    else:
        mesh = denseMesh
    model['output'] = mesh
    print('mesh: '+str(mesh))
    stuff
    return model

def buildModel(data,dataset,base,config):

    if isinstance(base,np.ndarray):
        vCount = len(base)
    else:
        vCount = base.get_shape().as_list()[-2]

    total = 0
    while 'image-'+str(total) in data:
        total += 1
    model = {}
    pose = data['pose']
    images = []
    usedVertsIter = iter(dataset.usedVerts)
    for i in range(total):
        #print('i: '+str(i))
        #print('data: '+str(data))
        if data['image-'+str(i)] is None or (hasattr(data['image-'+str(i)],'dtype') and data['image-'+str(i)].dtype == tf.string):
            continue
        usedVerts = next(usedVertsIter)
        with tf.variable_scope('dense-'+str(i)):
            if dataset.mask is not None:
                index = np.arange(dataset.mask.shape[1])[dataset.mask[i]]
                imageInput = tf.gather(pose,index,axis=1)
            else:
                imageInput = pose
            image = buildDenseModel(imageInput,len(usedVerts)*3,config)
            image = tf.reshape(image,(-1,len(usedVerts),3))
            model['dense'+str(i)] = image
            images.append(image)

    with tf.variable_scope('mesh_merge'):

        usedVerts = np.concatenate(dataset.usedVerts,0)
    #imgMesh = UVGenerator.mapImagesToMesh(images,dataset.usedUVs,dataset.usedVerts,base)
        unusedVerts = np.asarray([i for i in range(vCount) if i not in usedVerts])
        verts = np.concatenate((usedVerts,unusedVerts),0)
        if isinstance(base,np.ndarray):
            base = tf.constant(base[unusedVerts][np.newaxis],dtype=tf.float32)
            base = tf.tile(base,(tf.shape(pose)[0],1,1))
        else:
            base = tf.gather(base,unusedVerts,axis=-2)
        vs = tf.concat(images+[base],axis=-2)
        vOrder = np.argsort(verts)
        mesh = tf.gather(vs,vOrder,axis=-2)
    model['dense'] = images
    model['denseMesh'] = mesh
    if 'linear' in data:
        mesh = mesh + data['linear']
    model['output'] = mesh
    return model

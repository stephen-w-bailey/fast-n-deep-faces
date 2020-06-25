from collections import namedtuple
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import yaml

sys.path.append(os.path.join('..','cnnModel'))
sys.path.append('..')
import cnnModel
import UVGenerator

def buildModel(config,pose,addNeutral=True):

    with open(config['data_params']['cache_file'],'rb') as file:
        data = pickle.load(file)
    cache = data
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
    model['parts'] = parts
    model['cache'] = cache
    return model

def loadDefaultPose(rangeFile):

    default = []
    ranges = []
    with open(rangeFile) as file:
        for line in file:
            if ',' in line:
                line = line.strip().split(',')
            else:
                line = line.strip().split(' ')
            ranges.append([float(line[3]),float(line[4])])
            if len(line) < 6:
                default.append(0)
            else:
                default.append(float(line[5]))
    default = np.asarray(default)
    ranges = np.asarray(ranges)
    scale = 1/(ranges[:,1]-ranges[:,0]+1e-6)
    mask = (ranges[:,1]-ranges[:,0])<=1e-6
    scale[mask] = 1
    return default,scale

def buildLoss(data,model,config):
    
    # Build the approximation model
    with open(config['training_params']['approximation_config']) as file:
        approximationConfig = yaml.load(file)
    if 'base_config' in approximationConfig['model_params']:
        with open(approximationConfig['model_params']['base_config']) as file:
            baseConfig = yaml.load(file)
        baseModel = buildModel(baseConfig,model['output'])
        with tf.variable_scope('refine'):
            approxModel = buildModel(approximationConfig,model['output'],addNeutral=False)
        mesh = approxModel['output'] + baseModel['output']
        print('using refinement model')
    else:
        approxModel = buildModel(approximationConfig,model['output'])
        mesh = approxModel['output']
        print('not applying refinement')

    # Compute the points on the approximation
    with open(config['data_params']['point_file'],'rb') as file:
        pointData = pickle.load(file)
    with open(approximationConfig['data_params']['cache_file'],'rb') as file:
        configData = pickle.load(file)
    faces = configData['faces'][pointData['f']].reshape(-1)
    bary = pointData['b']
    faces = tf.reshape(tf.gather(mesh,faces,axis=1),(-1,len(bary),3,3))
    points = tf.reduce_sum(faces*bary[...,np.newaxis],2)

    # Compute the error
    losses = {}
    target = data['points']
    usedDims =[]
    dimMap = dict(x=0,y=1,z=2)
    for c in config['training_params']['compare_dims']:
        usedDims.append(dimMap[c])
    points = tf.gather(points,usedDims,axis=-1)
    target = tf.gather(target,usedDims,axis=-1)
    diff = tf.abs(points-target)
    error = tf.reduce_mean(diff)
    losses['dist_error'] = error

    default,scale = loadDefaultPose(config['data_params']['range_file'])
    reg = tf.reduce_mean(tf.abs(model['output']-default)*scale)
    alpha = float(config['training_params']['reg_weight'])
    losses['reg_error'] = alpha * reg

    if 'refine_reg' in config['training_params'] and 'base_config' in approximationConfig['model_params']:
        idx = np.arange(len(approxModel['parts']))[approxModel['parts']>=0]
        print('Using '+str(len(idx))+' vertices in refine regularization')
        v = tf.gather(approxModel['output'],idx,axis=1)
        reg = tf.reduce_mean(tf.square(v))
        alpha = float(config['training_params']['refine_reg'])
        losses['refine_reg'] = alpha*reg

    return losses,points,mesh

def buildOpt(loss,config):

    optVars = [v for v in tf.trainable_variables() if 'ik-model' in v.name]
    error = tf.add_n([loss[k] for k in loss])
    lr = tf.Variable(1e-3,trainable=False,name='lr')
    lrPH = tf.placeholder(tf.float32,(),name='lrPH')
    lrAssign = tf.assign(lr,lrPH)
    globalStep = tf.train.create_global_step()
    opt = tf.train.AdamOptimizer(lr)
    opt = opt.minimize(error,var_list=optVars,global_step=globalStep)
    ops = dict(lr=lr,lrPH=lrPH,lrAssign=lrAssign,
               opt=opt,globalStep=globalStep)
    return ops

def buildSummaries(data,model,loss):
    for error in loss:
        tf.summary.scalar('loss/'+error,loss[error])
    variables = [v for v in tf.trainable_variables() if 'ik-model' in v.name]
    with tf.variable_scope('l2_reg'):
        l2 = [tf.nn.l2_loss(v) for v in variables]
        weights = tf.add_n(l2)
        tf.summary.scalar('parameter_weights',weights)

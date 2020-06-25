import functools
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import tfplot

def figure_heatmap(heatmap, points=None, cmap='plasma', log=False):
    dims = len(heatmap.shape)
    if dims > 2:
        heatmap = heatmap[...,0]
    if log:
        heatmap = np.log(heatmap+1e-7)

    # draw a heatmap with a colorbar
    fig, ax = tfplot.subplots(figsize=(8, 8))       # DON'T USE plt.subplots() !!!!
    im = ax.imshow(heatmap, cmap=cmap)
    fig.colorbar(im)
    if points is not None:
        points = points*[heatmap.shape[1],heatmap.shape[0]]
        ax.plot(points[:,1],points[:,0],'.r',markersize=2)
    return fig

def l2Loss(x):
    return tf.reduce_sum(tf.square(x),axis=-1)

def l1Loss(x):
    return tf.reduce_sum(tf.abs(x),axis=-1)

def faceNormals(v,faces):

    with tf.variable_scope('face_normals'):
        
        # Compute the faces (vertex positions per triangle)
        f = np.asarray(faces).reshape(-1)
        face = tf.gather(v,f,axis=0)
        face = tf.reshape(face,(-1,3,3))

        # Compute the normals
        a = face[:,1]-face[:,0]
        b = face[:,2]-face[:,0]
        n = tf.cross(a, b)
        n = n / tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(n),axis=1)),(-1,1))

    return n

def vertexNormals(v,n,faces):

    with tf.variable_scope('vertex_normals'):
        maskIdx = np.arange(v.get_shape().as_list()[0])
        tvs = [[] for _ in range(len(maskIdx))]
        for i,f in enumerate(faces):
            for v in f:
                tvs[v].append(i)
        maxLen = np.max([len(tv) for tv in tvs])
        tvMat = np.zeros((len(tvs),maxLen)).astype('int32')
        tvMask = np.zeros(tvMat.shape).astype(bool)
        for i in range(tvMat.shape[0]):
            tvMat[i,:len(tvs[i])] = tvs[i]
            tvMask[i,:len(tvs[i])] = True

        # Create the computation graph to average the normals
        tvMat = tvMat.reshape(-1)
        tvMask = tvMask.reshape((-1,1)).astype('float32') # n*maxlen x 1
        n = tf.gather(n,tvMat,axis=0) * tvMask # n*maxlen x 3
        n = tf.reshape(n,(-1,maxLen,3))
        n = tf.reduce_mean(n,axis=1)
        n = n / tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(n),-1)),(-1,1))
        n = tf.reshape(n,(-1,3))
        
    return n

def buildLoss(data,model,config,usedVerts=None):

    # Vertex loss
    loss = {}
    y = data['mesh']
    yPred = model['output']
    diff = y-yPred
    if usedVerts is not None:
        diff = tf.gather(diff,usedVerts,axis=1)
    if config['training_params']['loss_function'] == 'l1':
        error = tf.reduce_mean(l1Loss(diff))
    elif config['training_params']['loss_function'] == 'l2':
        error = tf.reduce_mean(l2Loss(diff))
    else:
        raise ValueError('Unknown loss function '+str(config['training_params']['loss_function']))
    if 'vertex_loss' in config['training_params']['loss_weights']:
        weight = float(config['training_params']['loss_weights']['vertex_loss'])
    else:
        weight = 1
    loss['vertex_loss'] = weight * error

    # Normal loss
    if 'normal_loss' in config['training_params']['loss_weights']:
        faces = data['faces']
        def fn(v):
            n = faceNormals(v,faces)
            return n
        with tf.variable_scope('normal_loss'):
            vn = tf.map_fn(fn,y)
            vnPred = tf.map_fn(fn,yPred)
            if config['training_params']['normal_loss_function'] == 'l2':
                error = tf.reduce_mean(tf.square(vn-vnPred))
            elif config['training_params']['normal_loss_function'] == 'l1':
                error = tf.reduce_mean(tf.abs(vn-vnPred))
            else:
                raise NotImplementedError('Loss '+config['training_params']['normal_loss_function']+' not iplemented')
            weight = float(config['training_params']['loss_weights']['normal_loss'])
            loss['normal_loss'] = weight * error

    return loss

def buildOpt(loss,config,variables=None):
    error = tf.add_n([loss[k] for k in loss])
    lr = tf.Variable(1e-3,trainable=False,name='lr')
    lrPH = tf.placeholder(tf.float32,(),name='lrPH')
    lrAssign = tf.assign(lr,lrPH)
    globalStep = tf.train.create_global_step()
    opt = tf.train.AdamOptimizer(lr)
    opt = opt.minimize(error,global_step=globalStep,var_list=variables)
    ops = dict(lr=lr,lrPH=lrPH,lrAssign=lrAssign,
               opt=opt,globalStep=globalStep)
    return ops

def buildSummaries(data,model,loss,uvs=None):
    for error in loss:
        tf.summary.scalar('loss/'+error,loss[error])
    variables = tf.trainable_variables()
    l2 = [tf.nn.l2_loss(v) for v in variables]
    weights = tf.add_n(l2)
    tf.summary.scalar('parameter_weights',weights)
    if uvs is None:
        return
    keys = [k for k in model if 'cnn' in k]
    idx = [int(k[3:]) for k in keys]
    order = np.argsort(idx)
    for o,uv in zip(order,uvs):
        k = keys[o]
        image = model[k]
        axis = ['y']
        for i in range(len(axis)):
            tfplot.summary.plot(k+'/'+str(axis[i]),figure_heatmap,[image[0][...,i],uv])

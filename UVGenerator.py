import functools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def pdist2(x,y): # x is ...,m,d and y is ...,n,d
    rank = len(x.shape)
    transpose = [i for i in range(rank)]
    transpose[-1] = rank-2
    transpose[-2] = rank-1
    dist = -2*np.matmul(x,np.transpose(y,transpose)) # ...,m,n
    xx = np.sum(x**2,-1) # ...,m
    yy = np.sum(y**2,-1) # ...,n
    dist = dist + xx[...,np.newaxis] + np.expand_dims(yy,-2) # ...,m,n
    return np.maximum(dist,0)

def mapMeshToImage(v,uv,w,h):
    goodVerts = np.ones(len(uv),dtype=bool)
    uv = uv[goodVerts]
    idx = np.arange(len(goodVerts))[goodVerts]
    v = tf.gather(v,idx,axis=1)
    uvMin = [0,0]
    uvMax = [1,1]
    x=np.linspace(uvMin[1],uvMax[1],w)
    y=np.linspace(uvMin[0],uvMax[1],h)
    x,y=np.meshgrid(x,y)
    coord = np.stack([y,x],-1).astype('float32')
    coord = coord.reshape((-1,2))
    if isinstance(v,np.ndarray):
        if len(v.shape) == 2:
            v = v[np.newaxis]
        v = tf.constant(v)
    batchSize = v.get_shape().as_list()[0]
    dim = v.get_shape().as_list()[-1]
    uv = tf.tile(uv[np.newaxis],(batchSize,1,1))
    coord = np.repeat(coord[np.newaxis],batchSize,0)
    img = tf.contrib.image.interpolate_spline(uv,v,coord,1)
    img = tf.reshape(img,(batchSize,h,w,dim))
    return img,goodVerts

def mapImageToMesh(img,uv):
    uvMin = 0
    uvMax = 1
    uv = (uv-uvMin)/(uvMax-uvMin)
    h,w = img.get_shape().as_list()[-3:-1]
    uv = uv*[h-1,w-1]
    if isinstance(uv,np.ndarray):
        uv = uv.astype('float32')
    uv1,uv2 = uv[:,0],uv[:,1]
    uv1 = tf.clip_by_value(uv1,0,h-1)
    uv2 = tf.clip_by_value(uv2,0,w-1)
    uv = tf.stack([uv1,uv2],-1)
    x11 = tf.clip_by_value(tf.floor(uv),0,h-2) # This will cause problems if h != w
    x22 = x11 + 1
    x12 = tf.stack([x11[:,0],x22[:,1]],-1)
    x21 = tf.stack([x22[:,0],x11[:,1]],-1)
    img = tf.transpose(img,(1,2,0,3)) # h,w,b,c
    img11 = tf.transpose(tf.gather_nd(img,tf.cast(x11,tf.int32)),(1,0,2)) # b,n,c
    img12 = tf.transpose(tf.gather_nd(img,tf.cast(x12,tf.int32)),(1,0,2))
    img21 = tf.transpose(tf.gather_nd(img,tf.cast(x21,tf.int32)),(1,0,2))
    img22 = tf.transpose(tf.gather_nd(img,tf.cast(x22,tf.int32)),(1,0,2))
    a = (uv-x11)
    img1 = (1-tf.expand_dims(a[:,0],-1))*img11+tf.expand_dims(a[:,0],-1)*img21
    img2 = (1-tf.expand_dims(a[:,0],-1))*img12+tf.expand_dims(a[:,0],-1)*img22
    i = (1-tf.expand_dims(a[:,1],-1))*img1+tf.expand_dims(a[:,1],-1)*img2
    return i

def mapImagesToMesh(imgs,uvs,usedVerts,base):
    vs = []
    for img,uv,verts in zip(imgs,uvs,usedVerts):
        v = mapImageToMesh(img,uv)
        vs.append(v)
    usedVerts = np.concatenate(usedVerts,0)
    if isinstance(base,np.ndarray):
        vCount = len(base)
    else:
        vCount = base.get_shape().as_list()[-2]
    unusedVerts = np.asarray([i for i in range(vCount) if i not in usedVerts], dtype='int32')
    verts = np.concatenate((usedVerts,unusedVerts),0)
    if isinstance(base,np.ndarray):
        base = tf.constant(base[unusedVerts][np.newaxis],dtype=tf.float32)
        base = tf.tile(base,(tf.shape(vs[0])[0],1,1))
    else:
        base = tf.gather(base,unusedVerts,axis=-2)
    vs = tf.concat(vs+[base],axis=-2)
    vOrder = np.argsort(verts)
    mesh = tf.gather(vs,vOrder,axis=-2)
    return mesh

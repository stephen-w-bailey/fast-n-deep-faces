import numpy as np
import pickle
import tensorflow as tf

def rigidRegister(source, target):

    # Normalize the data
    v = target
    p = source
    pMean = np.mean(p,0)
    vMean = np.mean(v,0)
    pp = p - pMean
    vp = v - vMean

    # Compute the rotation
    M = vp.T.dot(pp)
    u,sig,v = np.linalg.svd(M)
    sig = 1 / sig
    Qsqrt = u.dot(np.diag(sig).dot(u.T))
    R = Qsqrt.dot(M).T

    # Compute the translation
    t = (vMean - pMean.dot(R)).astype('float32')

    return R,t

def rigidRegisterTF(source, target):

    # Normalize the data
    v = target
    p = source
    pMean = tf.reduce_mean(p,-2)
    vMean = tf.reduce_mean(v,-2)
    pp = p - pMean[:,np.newaxis]
    vp = v - vMean[:,np.newaxis]

    # Compute the rotation
    M = tf.matmul(tf.transpose(vp,(0,2,1)),pp)
    sig,u,v = tf.linalg.svd(M)
    sig = 1 / sig
    Qsqrt = tf.matmul(u,sig[...,np.newaxis]*tf.transpose(u,(0,2,1)))
    R = tf.transpose(tf.matmul(Qsqrt,M),(0,2,1))

    # Compute the translation
    t = vMean[:,np.newaxis]-tf.matmul(pMean[:,np.newaxis],R)

    return R,t

class RigidDeformer:

    def __init__(self,neutral,rigidFiles,mask):

        # Load the data
        self.neutral = neutral
        self.mask = mask
        self.usedNeutral = neutral[mask]
        self.parts = []
        rank = []
        for f in rigidFiles:
            with open(f,'rb') as file:
                data = pickle.load(file)
            self.parts += data['parts']
            rank += data['rank']

        # Choose the vertices for each part
        n = 10
        self.rank = []
        for r in rank:
            if len(r) == len(self.neutral):
                r = r[mask]
            order = np.argsort(r)
            self.rank.append(order[:n])

        # Create a list of all vertices not in parts
        parts = []
        for p in self.parts:
            parts += [i for i in p]
        parts = np.asarray(parts)
        self.notParts = np.asarray([i for i in range(len(neutral)) if i not in parts])
        parts = np.concatenate([parts,self.notParts],0)
        self.partIndex = np.argsort(parts)

    def deform(self,mesh):

        mesh = mesh.copy()
        usedMesh = mesh[self.mask]
        for r,p in zip(self.rank,self.parts):
            target = usedMesh[r]
            neutral = self.usedNeutral[r]
            rigid = self.neutral[p]
            R,t = rigidRegister(neutral,target)
            mesh[p] = rigid.dot(R)+t
        return mesh

    def deformTF(self,mesh):

        with tf.variable_scope('rigid_deform'):
            usedMesh = tf.gather(mesh,self.mask,axis=0)
            parts = []
            partIndex = []
            rs = np.concatenate(self.rank,0)
            neutral = self.usedNeutral[rs].reshape((len(self.rank),-1,3))
            target = tf.reshape(tf.gather(usedMesh,rs,axis=0),(len(self.rank),-1,3))
            R,t = rigidRegisterTF(neutral,target)
            maxP = np.max([len(p) for p in self.parts])
            rigid = np.zeros((len(self.parts),maxP,3),dtype='float32')
            index = []
            for i,p in enumerate(self.parts):
                rigid[i,:len(p)] = self.neutral[p]
                index.append(np.stack((i*np.ones(len(p),dtype='int32'),np.arange(len(p)).astype('int32')),-1))
            index = np.concatenate(index,axis=0)
            parts = tf.matmul(rigid,R)+t
            parts = tf.gather_nd(parts,index)
            parts = (parts,tf.gather(mesh,self.notParts,axis=0))
            parts = tf.concat(parts,axis=0)
            mesh = tf.gather(parts,self.partIndex,axis=0)
        return mesh

import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf

import UVGenerator

class DataLoader():

    def __init__(self,pg,sampleFile=None):
        self.pg = pg
        self.numV = np.sum(self.pg.active)
        pg.setPose(pg.defaultPose)
        self.neutral = pg.getVertices()[pg.active]

    def generator(self):
        cache = []
        maxCache = 10000
        while True:
            if len(cache) < maxCache:
                pose = self.pg.setRandomPose()
                mesh = self.pg.getVertices()[self.pg.active]
                cache.append((pose,mesh))
                if len(cache) == maxCache:
                    print('cache full')
            else:
                idx = np.random.choice(len(cache))
                pose,mesh = cache[idx]
            yield dict(pose=pose,mesh=mesh)

    def process(self,data):
        pose,mesh = data['pose'],data['mesh']
        pose.set_shape(self.pg.defaultPose.shape)
        mesh.set_shape((self.numV,3))
        return dict(pose=pose,mesh=mesh)

    def createDataset(self,batchsize):
        dataset = tf.data.Dataset.from_generator(self.generator,
                                                 dict(pose=tf.float32,
                                                      mesh=tf.float32))
        dataset = dataset.map(self.process)
        dataset = dataset.batch(batchsize)
        dataset = dataset.prefetch(5)
        iter = dataset.make_one_shot_iterator()
        element = iter.get_next()
        return element

class LinearDataLoader(DataLoader):

    def __init__(self,pg,linearModel):
        DataLoader.__init__(self,pg)

        with open(linearModel,'rb') as file:
            data = pickle.load(file)

        self.x = data['weights'].astype('float32')
        self.weights = data['ssdWeights'].astype('float32')
        self.restBones = data['ssdRestBones'].astype('float32')
        self.rest = data['ssdRest']
        self.k = data['k']
        
    def process(self,data):
        data = DataLoader.process(self,data)
        pose,mesh = data['pose'],data['mesh']
        pose = tf.concat((pose,tf.ones((1,))),0)
        bones = tf.reshape(tf.matmul(pose[np.newaxis],self.x),(self.k,4,3))
        approx = []
        for i in range(self.k):
            R = bones[i,:3]
            t = bones[i,3][np.newaxis]
            tRest = self.restBones[i,3]
            approx.append(tf.matmul(self.rest-tRest,R)+t)
        approx = tf.stack(approx,0)
        weights = self.weights.T[...,np.newaxis]
        approx = tf.reduce_sum(weights*approx,0)
        data = dict(pose=pose,mesh=mesh,linear=approx)
        return data

class ImageDataLoader(DataLoader):

    def __init__(self,pg,uvFile,linearModel=None,makeImages=False):
        DataLoader.__init__(self,pg)
        self.makeImages = makeImages
        if linearModel is not None:
            self.linearModel = LinearDataLoader(pg,linearModel)
        else:
            self.linearModel = None

        with open(uvFile,'rb') as file:
            data = pickle.load(file)

        self.faces = data['originalFaces']
        self.numV = np.max(self.faces)+1
        self.uv = data['uv'][:self.numV].astype('float32')
        self.uv = self.uv
        self.vCharts = data['vCharts'][:self.numV]
        if 'parameter_mask' in data:
            self.mask = data['parameter_mask']
        else:
            self.mask = None

    def process(self,data):
        data = DataLoader.process(self,data)
        mesh = data['mesh']
        self.usedVerts = []
        self.usedUVs = []
        
        if self.linearModel is not None:
            data = self.linearModel.process(data)
        else:
            data['linear'] = self.neutral
        mesh = mesh - data['linear']

        for i in range(np.max(self.vCharts)+1):
            idx = np.arange(self.numV)[self.vCharts==i]
            if len(idx) == 0:
                data['image-'+str(i)] = 'empty'
                continue
            ref = self.faces.reshape(-1)
            usedFaces = [True if v in idx else False for v in ref]
            usedFaces = np.sum(np.asarray(usedFaces).reshape((-1,3)),-1) > 0
            faceIdx = np.arange(len(self.faces))[usedFaces]
            idx = np.arange(len(self.vCharts))[self.vCharts==i]
            if len(idx) == 0:
                raise ValueError('Chart index '+str(i)+' has no assigned verties')
            meshPart = tf.gather(mesh,idx)
            image,usedVerts = UVGenerator.mapMeshToImage(meshPart[np.newaxis],self.uv[idx],self.imageSize,self.imageSize)
            if not self.makeImages:
                image = tf.zeros((self.imageSize,self.imageSize,3))
            self.usedUVs.append(self.uv[idx[usedVerts]])
            self.usedVerts.append(idx[usedVerts])
            image = image[0]
            data['image-'+str(i)] = image

        return data

    def createDataset(self,batchsize,imageSize):
        self.imageSize = imageSize
        return DataLoader.createDataset(self,batchsize)


class AnimationLoader(ImageDataLoader):
    
    def __init__(self,pg,animData,uvFile,linearModel=None,fixToRange=False):
        ImageDataLoader.__init__(self,pg,uvFile,linearModel)

        newAnim = animData.copy()
        if fixToRange:
            for i,node in enumerate(pg.nodes):
                node = [n for n in node] # Copy the data so modification won't change the original
                frac = 0.1*(node[3]-node[2])
                node[2] += frac
                node[3] -= frac
                if np.any(newAnim[:,i]<node[2]):
                    print('Found '+str(np.sum(newAnim[:,i]<node[2]))+' values for '+str(node[:2])+' below '+str(node[2]))
                if np.any(newAnim[:,i]>node[3]):
                    print('Found '+str(np.sum(newAnim[:,i]>node[3]))+' values for '+str(node[:2])+' above '+str(node[3]))
                newAnim[:,i] = np.minimum(np.maximum(newAnim[:,i],node[2]),node[3])
            diff = np.sum(np.square(newAnim-animData),1)
            print('Clamped values in '+str(np.sum(diff>0))+' frames')
            animData = newAnim
            print('Checking if animation was correctly modified')
            for i,node in enumerate(pg.nodes):
                node = [n for n in node] # Copy the data so modification won't change the original
                frac = 0.1*(node[3]-node[2])
                node[2] += frac
                node[3] -= frac
                if np.any(animData[:,i]<node[2]):
                    print('Found '+str(np.sum(animData[:,i]<node[2]))+' values for '+str(node[:2])+' below '+str(node[2]))
                if np.any(animData[:,i]>node[3]):
                    print('Found '+str(np.sum(animData[:,i]>node[3]))+' values for '+str(node[:2])+' above '+str(node[3]))

        self.animData = animData

    def generator(self):
        for d in self.animData:
            self.pg.setPose(d)
            mesh = self.pg.getVertices()[self.pg.active]
            yield dict(pose=d,mesh=mesh)

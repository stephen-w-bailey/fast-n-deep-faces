import numpy as np
import pickle
import tensorflow as tf

class DataLoader():

    def __init__(self,sampleFile,pointFile,fullyRandom=False,noise=None):

        with open(sampleFile,'rb') as file:
            data = pickle.load(file)
        if not isinstance(data['samples'],np.ndarray):
            data['samples'] = np.asarray(data['samples'])
        self.samples = data['samples'].astype('float32')
        if 'bins' in data:
            bins = data['bins']
            self.bins = [[] for _ in range(np.max(bins)+1)]
            for i in range(len(bins)):
                self.bins[bins[i]].append(i)
        else:
            self.bins = None

        with open(pointFile,'rb') as file:
            data = pickle.load(file)
            self.points = data['points'].astype('float32')

        self.fullyRandom = fullyRandom
        self.noise = noise

    def generator(self):
        
        while True:
            if self.fullyRandom:
                pointCount = self.points.shape[1]
                if self.bins is not None:
                    binIdx = np.random.choice(len(self.bins))
                    sampleIdx = [self.bins[binIdx][i] for i in np.random.choice(len(self.bins[binIdx]),pointCount)]
                    sampleIdx = np.asarray(sampleIdx)
                else:
                    sampleIdx = np.random.choice(len(self.samples),pointCount)
                points = self.points[sampleIdx,np.arange(pointCount)]
                sample = self.samples[sampleIdx[0]]
                yield dict(pose=sample,points=points)
            else:
                if self.bins is not None:
                    binIdx = np.random.choice(len(self.bins))
                    sampleIdx = self.bins[binIdx][np.random.choice(len(self.bins[binIdx]))]
                else:
                    sampleIdx = np.random.choice(len(self.samples))
                sample = self.samples[sampleIdx]
                points = self.points[sampleIdx]
                if self.noise is not None:
                    points = points + np.random.uniform(-self.noise,self.noise,points.shape).astype('float32')
                yield dict(pose=sample,points=points)

    def process(self,data):
        pose,points = data['pose'],data['points']
        pose.set_shape(self.samples.shape[1])
        points.set_shape((self.points.shape[1],3))
        return data

    def createDataset(self,batchSize):
        dataset = tf.data.Dataset.from_generator(self.generator,
                                                 dict(pose=tf.float32,
                                                      points=tf.float32))
        dataset = dataset.map(self.process)
        dataset = dataset.batch(batchSize)
        dataset = dataset.prefetch(5)
        iter = dataset.make_one_shot_iterator()
        element = iter.get_next()
        return element

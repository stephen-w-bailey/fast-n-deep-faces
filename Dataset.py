import numpy as np
import pickle

class DataIterator:

    def __init__(self, dataset, batchSize):
        self.dataset = dataset
        length = len(dataset.data[list(dataset.data.keys())[0]])
        self.index = np.arange(length)
        np.random.shuffle(self.index)
        self.ptr = 0
        self.batchSize = batchSize

    def __next__(self):
        if self.ptr == len(self.index):
            raise StopIteration
        end = self.ptr + self.batchSize
        end = min(end,len(self.index))
        data = {}
        idx = self.index[self.ptr:end]
        for key in self.dataset.data:
            d = self.dataset.data[key][idx]
            if key in self.dataset.mean:
                d = (d-self.dataset.mean[key])/self.dataset.std[key]
            data[key] = d
        self.ptr = end
        return data

class Dataset:

    def __init__(self,genFn,processFn):
        self.genFn = genFn
        self.processFn = processFn
        self.data = None
        self.batchSize = 1
        self.mean = {}
        self.std = {}

    def setBatchSize(self,size):
        self.batchSize = size

    def generateData(self,n):
        data = self.genFn(n)
        #data = self.processFn(data)
        if self.data is None:
            self.data = data
        else:
            for key in self.data:
                self.data[key] = np.concatenate([self.data[key],data[key]],0)

    def processData(self):
        data = {}
        for key in self.data:
            if key in self.mean:
                data[key] = (self.data[key]-self.mean[key])/self.std[key]
            else:
                data[key] = self.data[key]
        newData = self.processFn(data)
        for key in newData:
            if key not in self.data:
                self.data[key] = newData[key]

    def normalize(self,key):
        data = self.data[key]
        r = [i for i in range(len(data.shape)-1)]
        r = tuple(r)
        mean = np.mean(data,r)
        std = np.std(data,r)
        self.mean[key] = mean
        self.std[key] = std

    def clear(self):
        self.data = None

    def saveData(self,fileName):
        with open(fileName,'wb') as file:
            pickle.dump(self.data,file,protocol=-1)

    def loadData(self,fileName):
        with open(fileName,'rb') as file:
            self.data = pickle.load(file)

    def saveMoments(self,fileName):
        with open(fileName,'wb') as file:
            pickle.dump(dict(mean=self.mean,std=self.std),file)

    def addMomentsToDict(self,d):
        d['mean'] = self.mean
        d['std'] = self.std

    def momentsFromDict(self,d):
        self.mean = d['mean']
        self.std = d['std']

    def __iter__(self):
        return DataIterator(self,self.batchSize)

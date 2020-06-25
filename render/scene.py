import numpy as np

from . import camera

class Scene:

    def __init__(self):
        self.meshes = []
        self.camera = camera.TrackBall()
        self.amb = np.asarray([0.1,0.1,0.1]).astype('float32')
        self.direct = np.asarray([0.8,0.8,0.8]).astype('float32')
        #self.directDir = np.asarray([0.0,0.0,-1.0]).astype('float32')

    def addMesh(self,mesh):
        self.meshes.append(mesh)

    def setCamera(self,camera):
        self.camera = camera

    @property
    def directDir(self):
        direction = self.camera.pose[:3,:3].T.dot(self.camera.pose[3,:3])
        return -direction

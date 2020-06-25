import numpy as np
import trimesh

class Camera:

    def __init__(self):
        self.cop = np.asarray([0,0]).astype('float32')
        self.fov = 45
        self.setAspect(1)
        self.pose = np.eye(4)
        self.projFunc = self.perspProj

    def getTransform(self):
        R = np.eye(3)
        t = np.asarray([0,0,1])
        return R,t

    def setAspect(self,aspect):
        self.a = aspect

    def setProjection(self,typeName):
        if typeName.lower() == 'perspective':
            self.projFunc = self.perspProj
        if typeName.lower() == 'orthographic':
            self.projFunc = self.orthoProj
        else:
            raise ValueError('Cannot handle projection of type '+str(typeName))

    def getRay(self,screen):
        if self.projFunc == self.perspProj:
            raise NotImplementedError('Ray casting through perspective projection not implemented')
        else:
            x,y = screen[:2]
            s = self.pose[3,2] * 1
            x = x*s/2
            y = y*s/2
            p = np.asarray([x,y,0])
            r = np.asarray([0,0,-1])
            return p,r

    def orthoProj(self):
        ortho = np.zeros((4,4))
        s = self.pose[3,2] * 1
        f = 1000
        ortho[0,0] = 2/s
        ortho[1,1] = 2/s/self.a
        ortho[2,2] = -2/f
        ortho[3,3] = 1
        return ortho

    def perspProj(self):
        persp = np.zeros((4,4))
        nearVal = 0.1
        farVal = 200.0
        A = -(farVal+nearVal)/(farVal-nearVal)
        B = -farVal*nearVal/(farVal-nearVal)
        focal = 1/np.tan(self.fov/2*np.pi/180)
        persp[0,:] = np.asarray([focal,0,0,0])
        persp[1,:] = np.asarray([0,focal/self.a,0,0])
        persp[2,:] = np.asarray([0,0,A,B])
        persp[3,:] = np.asarray([0,0,-1,0])
        return persp

    def projMat(self):
        R,t = self.getTransform()
        T = np.eye(4)
        T[:3,3] = -t
        mat = np.eye(4)
        mat[:3,:3] = R
        #mat = mat.dot(T)
        mat = T.dot(mat)
        #T = np.eye(4)
        #T[:3,3] = np.asarray([self.cop[0],self.cop[1],0])
        #mat = T.dot(mat)

        proj = self.projFunc()

        #print('mvp matrix:\n'+str(persp.dot(mat)))
        return proj.dot(mat),R.T,t
        #return mat.dot(persp),R.T,t

class TrackBall(Camera):

    def __init__(self):
        Camera.__init__(self)
        self.target = np.asarray([0,0,0]).astype('float32')
        self.pose = np.eye(4)
        self.pose[3,:3] = np.asarray([0,0,5])

    def down(self, point):
        self.downPoint = np.asarray(point).astype('float32')
        self.prevPose = self.pose.copy()

    def drag(self, point):
        target = self.target
        x = self.prevPose[:3,0].flatten()
        y = self.prevPose[:3,1].flatten()
        z = self.prevPose[:3,2].flatten()
        eye = self.prevPose[:3,3].flatten()
        point = np.asarray(point).astype('float32')
        dx,dy = point-self.downPoint

        xang = dx/100
        yang = dy/100
        xRot = trimesh.transformations.rotation_matrix(xang,y,target)
        yRot = trimesh.transformations.rotation_matrix(yang,x,target)
        self.pose = yRot.dot(xRot.dot(self.prevPose))

        x = self.pose[:3,0].flatten()
        y = self.pose[:3,1].flatten()
        z = self.pose[:3,2].flatten()

    def zoom(self, point):
        t = self.prevPose[3,:3].copy()
        point = np.asarray(point).astype('float32')
        dx,dy = point-self.downPoint
        t[2] += dy/50
        self.pose[3,:3] = t

    def getTransform(self):
        R = self.pose[:3,:3]
        t = self.pose[3,:3]
        return R,t

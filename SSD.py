import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.optimize import lsq_linear
import tqdm

limit = 1000

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
    if np.linalg.det(R) < 0:
        print('det(R): '+str(np.linalg.det(R)))
        print('rigidRegister: Error, not a rotation')

    # Compute the translation
    t = (vMean - pMean.dot(R)).astype('float32')

    return R,t

# Implementation of the paper "Smooth Skinning Decomposition with Rigid Bones"
class SSD:

    def __init__(self):
        pass

    def initialize(self, rest, meshes, k, iterations=20, faces=None):

        # Create initial segments through k-means
        whitened = whiten(rest)
        codebook,distortion = kmeans(whitened,k)
        assignment,dist = vq(whitened,codebook)

        # Create initial bone configurations
        m = len(meshes)
        v = len(rest)
        restBones = np.zeros((k,4,3))
        for i in range(k):
            restBones[i,:3] = np.eye(3)
            restBones[i,3] = np.mean(rest[assignment==i],0)
        bones = np.zeros((m,k,4,3))
        restParts = [rest[assignment==i] - np.mean(rest[assignment==i],0) for i in range(k)]

        def computeRigidBones(assignment):
            for i in range(m):
                for j in range(k):
                    if np.sum(assignment==j) < 3:
                        raise RuntimeError('Encountered bone with less than 3 vertices assigned')
                    part = meshes[i,assignment==j]
                    r,t = rigidRegister(restParts[j],part)
                    bones[i,j,:3] = r
                    bones[i,j,3] = t
            return bones
        bones = computeRigidBones(assignment)

        # Loop a few times
        for _ in tqdm.trange(iterations):

            # Approximate all vertices with all bones
            approx = np.zeros((k,m,v,3))
            for i in range(k):
                R = bones[:,i,:3] # m x 3 x 3
                t = bones[:,[i],3] # m x 1 x 3
                tRest = restBones[i,3] # 3
                approx[i] = np.transpose(np.dot(np.transpose(R,(0,2,1)),(rest-tRest).T),(0,2,1))+t

            # Assign each vertex to the bone that best approximates it
            diff = np.mean(np.square(approx-meshes),(1,3)) # k x v
            assignment = np.argmin(diff,0)

            if faces is not None:
                sums = np.asarray([np.sum(assignment==i) for i in range(k)])
                while any(sums<3):
                    idx = list(sums<3).index(True)
                    f = np.random.choice(len(faces))
                    assignment[faces[f]] = idx
                    sums = np.asarray([np.sum(assignment==i) for i in range(k)])

            # Update the bones
            for i in range(k):
                restBones[i,3] = np.mean(rest[assignment==i],0)
            restParts = [rest[assignment==i] - np.mean(rest[assignment==i],0) for i in range(k)]
            bones = computeRigidBones(assignment)
                
        # Save the results
        self.weights = np.zeros((v,k))
        self.weights[range(v),assignment] = 1
        self.restBones = restBones
        self.rest = rest
        self.bones = bones
        self.meshes = meshes

    # Fix the bones and compute the vertex weights with affinity k
    def computeWeights(self,k=4):
        
        numV = len(self.rest)
        numB = len(self.restBones)
        k = min(k,numB)
        T = len(self.meshes)
        for i in range(numV):
            # Build the least squares problem
            A = np.zeros((3*T,numB))
            b = self.meshes[:,i].reshape(-1) # T*3

            R = self.bones[:,:,:3] # T x numB x 3 x 3
            t = self.bones[:,:,3] # T x numB x 3
            v = self.rest[i]-self.restBones[:,3] # numB x 3

            for j in range(numB):
                Rv = np.sum(R[:,j].reshape((-1,3,3,1))*v[j].reshape((-1,3,1,1)),-3) # T x 3 x 1
                Rvt = Rv.reshape((-1,3)) + t[:,j] # T x 3
                A[:,j] = Rvt.reshape(-1)

            # Solve the least squares problem
            bounds = (0,1)
            res = lsq_linear(A,b,bounds,method='bvls')
            w = res.x
            w = w / np.sum(w) # Fix any small numerical inaccuracies

            # Find the k best weights
            effect = np.sum(np.square(A),0)*np.square(w)
            indices = np.argpartition(effect,numB-k)[numB-k:]
            A = A[:,indices]
            res = lsq_linear(A,b,bounds,method='bvls')
            newW = res.x
            newW = newW / np.sum(newW)
            
            self.weights[i] = 0
            self.weights[i][indices] = newW

    def computeBones(self,meshes=None,bones=None):

        if meshes is None:
            meshes = self.meshes
            bones = self.bones
        elif bones is None:
            raise ValueError('SSD::computeBones: New mesh provided without bones')
        bones = bones.copy()

        # Divide dataset to avoid memory errors
        if len(bones) > limit:
            count = len(bones)
            bones1 = self.computeBones(meshes[:count//2],bones[:count//2])
            bones2 = self.computeBones(meshes[count//2:],bones[count//2:])
            bones = np.concatenate((bones1,bones2),0)
            return bones

        # Update the bones one at a time
        numB = len(self.restBones)
        T = len(meshes)
        B = range(numB)
        p = self.rest-self.restBones[:,3].reshape((-1,1,3)) # numB x v x 3
        for b in range(numB):

            # Remove the residual (Equation 6)
            others = list(B)
            del others[others.index(b)]
            R = bones[:,others][:,:,:3] # T x numB-1 x 3 x 3
            t = bones[:,others][:,:,3] # T x numB-1 x 3
            v = p[others].transpose((0,2,1)) # numB-1 x 3 x v
            q = meshes.copy() # T x v x 3
            for j in range(len(others)):
                Rv = np.sum(R[:,j].reshape((-1,3,3,1))*v[j].reshape((1,3,1,-1)),-3) # T x 3 x v
                Rv = Rv.transpose((0,2,1)) # T x v x 3
                Rvt = Rv + t[:,j][:,np.newaxis] # T x v x 3
                q -= self.weights[:,others[j]].reshape((1,-1,1)) * Rvt

            # Compute the remaining deformation
            rest = p[b]
            pStar = np.sum(np.square(self.weights[:,b])[...,np.newaxis]*rest,0) # v x 3 (Equation 8)
            pStar = pStar/np.sum(np.square(self.weights[:,b]))
            pBar = rest - pStar
            qStar = np.sum(self.weights[:,b][...,np.newaxis]*q,1)/np.sum(np.square(self.weights[:,b])) # T x 3
            qBar = q - self.weights[:,b][...,np.newaxis]*qStar.reshape((-1,1,3))
            P = self.weights[:,b][...,np.newaxis]*pBar # v x 3
            P = P.T # 3 x v
            QT = qBar # T x v x 3
            PQT = np.transpose(np.dot(np.transpose(QT,(0,2,1)),P.T),(0,2,1)) # T x 3 x 3
            try:
                u,_,v = np.linalg.svd(PQT)
            except np.linalg.linalg.LinAlgError:
                print('SVD error on the following matrix: '+str(PQT))
                print('QT[0]: '+str(QT[0]))
                print('P[0]: '+str(P[0]))
                raise
                
            u = u.transpose((0,2,1))
            R = np.sum(v.reshape((-1,3,3,1))*u.reshape((-1,3,1,3)),-3)
            t = qStar-R.transpose((0,1,2)).dot(pStar)
            bones[:,b,:3] = R.transpose((0,2,1))
            bones[:,b,3] = t

        return bones

    def runSSD(self,rest,meshes,numBones,k=4,faces=None):
        print('Initializing:')
        self.initialize(rest,meshes,numBones,faces=faces)

        maxIter = 20
        error = self.getFitError()
        eps = 1e-8
        print('Initial error: '+str(error))
        for _ in range(maxIter):
            self.computeWeights(k=k)
            self.bones = self.computeBones()
            newError = self.getFitError()
            print('New error: '+str(newError))
            if newError > error-eps:
                break
            error = newError

    def fitBonesToMesh(self,meshes):

        # Fit the bones first for rigid skinning
        m = len(meshes)
        k = self.bones.shape[1]
        bones = np.zeros((m,k,4,3))
        bones[:,:,:3] = np.eye(3)
        assignment = np.argmax(self.weights,1)
        restParts = [self.rest[assignment==i] - np.mean(self.rest[assignment==i],0) for i in range(k)]
        for i in range(m):
            for j in range(k):
                part = meshes[i,assignment==j]
                r,t = rigidRegister(restParts[j],part)
                bones[i,j,:3] = r
                bones[i,j,3] = t

        initialError = self.getFitError(meshes,bones)
        print('Rigid fit error: '+str(initialError))

        maxItr = 10
        eps = 1e-4
        for i in range(maxItr):
            bones = self.computeBones(meshes,bones)
            error = self.getFitError(meshes,bones)
            print('Fit error: '+str(error))
            if error > initialError-eps:
                break
            initialError = error

        return bones

    def getFitError(self,meshes=None,bones=None,returnMean=True):

        if meshes is None:
            meshes = self.meshes
            bones = self.bones
        elif bones is None:
            raise ValueError('SSD::getFitError: New mesh provided without bones')

        # Divide dataset to avoid memory errors
        if len(bones) > limit:
            count = len(bones)
            diff1 = self.getFitError(meshes[:count//2],bones[:count//2],False)
            diff2 = self.getFitError(meshes[count//2:],bones[count//2:],False)
            diff = np.concatenate((diff1,diff2),0)
            if returnMean:
                return np.mean(diff)
            else:
                return diff
        
        # Rigidly transform by every bone (not that efficient)
        k = self.weights.shape[1]
        v = self.rest.shape[0]
        T = len(bones)
        approx = np.zeros((T,k,v,3))
        for i in range(k):
            R = bones[:,i,:3] # T x 3 x 3
            t = bones[:,[i],3] # T x 1 x 3
            tRest = self.restBones[i,3] # 3
            vR = R.transpose((0,2,1)).dot((self.rest-tRest).T).transpose((0,2,1))
            approx[:,i] = vR+t
            
        weights = self.weights.T[...,np.newaxis] # k x v x 1
        approx = np.sum(weights*approx,1)
        diff = np.sqrt(np.sum(np.square(approx-meshes),-1))
        
        if returnMean:
            return np.mean(diff)
        else:
            return diff

    def computeMesh(self,bones):

        # Rigidly transform by every bone (not that efficient)
        k = self.weights.shape[1]
        v = self.rest.shape[0]
        approx = np.zeros((k,v,3))
        for i in range(k):
            R = bones[i,:3] # 3 x 3
            t = bones[[i],3] # 1 x 3
            tRest = self.restBones[i,3] # 3
            approx[i] = (self.rest-tRest).dot(R)+t

        # Apply blending
        weights = self.weights.T[...,np.newaxis] # k x v x 1
        approx = np.sum(weights*approx,0) # v x 3
        return approx

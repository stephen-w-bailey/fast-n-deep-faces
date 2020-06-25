import logging
try:
    import maya.api.OpenMaya as om
    import pymel
    import pymel.core
    usingMaya = True
except:
    logging.warning('PoseGenerator not running in maya')
    usingMaya = False
import functools
import numpy as np
import random
import socket
import struct
import time

class PoseGenerator(object):

    def __init__(self,controlFile,geoFile):
        self.loadControls(controlFile)
        self.loadGeoNodes(geoFile)
        self.sampler = None

    # Read a file specifying the rig controls and the range in which
    # they should be randomly set
    # Each line is as follows:
    # ControlName ControlType Index MinVAlue MaxValue
    # Control type is either r-rotation, t-translation, or s-scale
    # Index is either x, y, or z
    def loadControls(self,fileName):
        self.nodes = {}
        self.numControls = 0
        self.nameOrder = []
        self.defaultPose = []
        self.poseRange = []
        self._active = None
        with open(fileName) as file:
            for line in file:
                args = line.split(' ')
                nodeName = args[0]
                if len(args) < 6:
                    self.defaultPose.append(0) # Assume that the default value is 0 if not provided
                else:
                    self.defaultPose.append(float(args[5]))
                if nodeName not in self.nodes:
                    if usingMaya:
                        node = pymel.core.general.PyNode(nodeName)
                    else:
                        node = None
                    self.nodes[nodeName] = [node]
                    self.nameOrder.append(nodeName)
                controlType = args[1]
                if args[2].isdigit():
                    index = int(args[2])
                else:
                    coords = ['x','y','z']
                    if args[2] in coords:
                        index = args[2]
                    else:
                        if usingMaya:
                            controlNode = pymel.core.general.PyNode(nodeName+'.'+args[2])
                            index = controlNode
                        else:
                            controlNode = None
                            index = -1
                minValue = float(args[3])
                maxValue = float(args[4])
                self.nodes[nodeName].append((controlType,index,minValue,maxValue))
                self.poseRange.append((minValue,maxValue))
                self.numControls += 1
        self.defaultPose = np.asarray(self.defaultPose)

    def loadGeoNodes(self,geoFile):
        if not usingMaya:
            return
        
        self.geoNodes = []
        self.omGeoNodes = []
        self.geoNames = []
        self.useFull = []
        with open(geoFile) as file:
            for line in file:
                name = line.strip()
                if name[-1] == '*':
                    self.useFull.append(True)
                    name = name[:-1]
                else:
                    self.useFull.append(False)
                self.geoNames.append(name)
                self.geoNodes.append(pymel.core.general.PyNode(name))
                self.geoNodes[-1].select()
                selectionLs = om.MGlobal.getActiveSelectionList()
                selObj = selectionLs.getDagPath(0)   # index 0 in selection
                self.omGeoNodes.append(om.MFnMesh(selObj))

    def setSampler(self,sampler):
        self.sampler = sampler

    def createRandomPose(self):
        if self.sampler is None:
            pose = []
            for name in self.nameOrder:
                nodeList = self.nodes[name]
                for control in nodeList[1:]:
                    controlType,index,minValue,maxValue = control
                    val = random.random()*(maxValue-minValue)+minValue
                    pose.append(val)
            return pose
        else:
            pose = self.sampler.getRandomPose()
            return pose
        
    def setRandomPose(self):
        pose = self.createRandomPose()
        self.setPose(pose)
        return pose

    def getControls(self,attrs):
        res = {}
        for attr in attrs:
            res[attr] = attr.get()
        return res

    def setControls(self,attrs):
        for attr in attrs:
            attr.set(attrs[attr])
        
    def setPose(self,pose):
        if not usingMaya:
            return
        
        indexMap = {'x':0,'y':1,'z':2}
        pose = [float(p) for p in pose]
        for i in range(len(pose)):
            pose[i] = min(max(pose[i],self.poseRange[i][0]),self.poseRange[i][1])
        pose = iter(pose)
        for name in self.nameOrder:
            nodeList = self.nodes[name]
            node = nodeList[0]
            attrs = [i[1] for i in nodeList[1:] if not isinstance(i[1],str)]
            nodeSetFunctions = {'r':node.setRotation if hasattr(node,'setRotation') else None,
                                't':node.setTranslation if hasattr(node,'setTranslation') else None,
                                'c':self.setControls}
            nodeGetFunctions = {'r':node.getRotation if hasattr(node,'getRotation') else None,
                                't':node.getTranslation if hasattr(node,'getTranslation') else None,
                                'c':functools.partial(self.getControls,attrs)}
            controls = {}
            for control in nodeList[1:]:
                controlType,index,minValue,maxValue = control
                val = next(pose)
                if controlType not in controls:
                    controls[controlType] = nodeGetFunctions[controlType]()
                controls[controlType][indexMap[index] if isinstance(index,str) else index] = val
            for control in controls:
                nodeSetFunctions[control](controls[control])
                
    def getVertices(self,index=None):
        verts = []
        if not usingMaya:
            return None

        if index is None:
            for mesh,node in zip(self.omGeoNodes,self.geoNodes):
                points = mesh.getPoints()
                v = np.array(points)[:,:3]
                T = np.array(node.getMatrix())
                R = T[:3,:3]
                t = T[3,:3]
                v = v.dot(R)+t
                verts.append(v)
        else:
            mesh = self.omGeoNodes[index]
            points = mesh.getPoints()
            verts.append(np.array(points)[:,:3])
        return np.concatenate(verts,0)

    # Returns a boolean mask for each control indicating if the control moves
    # each vertex in the mesh
    # n is the number of samples to make per control to estimate the influence
    def getControlInfluence(self,n=16):
        if self._active is None:
            self.computeActiveVertices()
        mesh = self.getVertices()[self.active]
        masks = np.zeros((self.numControls,len(mesh)),dtype='bool')

        # Run through each control separately
        eps = 1e-6
        for i in range(self.numControls):
            p = np.stack([self.createRandomPose() for _ in range(n+1)],0)
            self.setPose(p[0])
            base = self.getVertices()[self.active]
            newP = np.repeat(p[[0]],n,0)
            newP[:,i] = p[1:,i]
            meshes = []
            for pose in newP:
                self.setPose(pose)
                meshes.append(self.getVertices()[self.active])
            meshes = np.stack(meshes,0)
            diff = np.sum(np.square(base-meshes),-1)
            diff = np.mean(diff,0)
            masks[i] = diff>eps
        return masks

    def getEdges(self,useActive=True):

        if len(self.geoNames) > 1:
            raise NotImplementedError('Cannot get edges for multiple meshes')
        
        verts = self.getVertices()
        idx = np.arange(len(verts))
        idxMap = idx.copy()
        if useActive:
            if not hasattr(self,'active'):
                self.computeActiveVertices()
            idx = idx[self.active]
            idxMap[:] = -1
            idxMap[self.active] = np.arange(len(idx))
        logging.info('Finding edges from '+str(len(idx))+' vertices')

        # Get the vertices
        baseName = self.geoNames[0]+'.vtx'
        vs = []
        for i in idx:
            name = baseName+'['+str(i)+']'
            vs.append(pymel.core.general.PyNode(name))

        # Get the edges
        es = pymel.core.modeling.polyListComponentConversion(vs,fv=True,te=True)
        es = [pymel.core.general.PyNode(edge) for edge in es]
            
        # Get the connected verties
        e = []
        for edges in es:
            itr = iter(edges)
            for edge in itr:
                a,b = edge.connectedVertices()
                a,b = idxMap[a.index()],idxMap[b.index()]
                if a == -1 or b == -1:
                    continue
                e.append((min(a,b),max(a,b)))
        return e

    def getFacesOnMesh(self,meshIdx,useActive=True):
        
        #if len(self.geoNames) > 1:
        #    raise NotImplementedError('Cannot get edges for multiple meshes')

        verts = self.getVertices(meshIdx)
        idx = np.arange(len(verts))
        idxMap = idx.copy()
        if useActive:
            if self._active is None:
                self.computeActiveVertices()
            idx = idx[self._active[meshIdx]]
            idxMap[:] = -1
            idxMap[self._active[meshIdx]] = np.arange(len(idx))
        logging.info('Finding faces from '+str(len(idx))+' vertices')

        # Get the vertices
        baseName = self.geoNames[meshIdx]+'.vtx'
        vs = []
        for i in idx:
            name = baseName+'['+str(i)+']'
            vs.append(pymel.core.general.PyNode(name))

        # Get the faces
        fs = pymel.core.modeling.polyListComponentConversion(vs,fv=True,tf=True)
        fs = [pymel.core.general.PyNode(face) for face in fs]

        # Get the vertices on the faces
        f = []
        for faces in fs:
            itr = iter(faces)
            for face in itr:
                vs = face.getVertices()
                vs = [idxMap[v] for v in vs]
                if any([v==-1 for v in vs]):
                    continue # Found a face with a vertex not being used
                if len(vs) == 3:
                    f.append(vs)
                elif len(vs) == 4:
                    f.append([vs[0],vs[1],vs[2]])
                    f.append([vs[2],vs[3],vs[0]])
                else:
                    logging.warning('Face with '+str(len(vs))+' vertices encountered, ignoring face')
        return f

    def getFaces(self,useActive=True):

        fs = []
        if self._active is None:
            self.computeActiveVertices()
        for i in range(len(self.geoNames)):
            f = np.asarray(self.getFacesOnMesh(i,useActive))
            fs.append(f)
        fullF = []
        count = 0
        for f,a in zip(fs,self._active):
            if len(f) == 0:
                continue
            fullF.append(f+count)
            if useActive:
                count += np.sum(a)
            else:
                count += len(a)
        f = np.concatenate(fullF,0)
        return f

    def getUVIndexOnMesh(self,meshIndex,useActive=True):
        
        #if len(self.geoNames) > 1:
        #    raise NotImplementedError('Cannot get edges for multiple meshes')

        verts = self.getVertices(meshIndex)
        idx = np.arange(len(verts))
        idxMap = idx.copy()
        if useActive:
            if self._active is None:
                self.computeActiveVertices()
            idx = idx[self._active[meshIndex]]
            idxMap[:] = -1
            idxMap[self._active[meshIndex]] = np.arange(len(idx))
        logging.info('Finding faces from '+str(len(idx))+' vertices')

        # Get the vertices
        baseName = self.geoNames[meshIndex]+'.vtx'
        vs = []
        for i in idx:
            name = baseName+'['+str(i)+']'
            vs.append(pymel.core.general.PyNode(name))

        # Get the faces
        fs = pymel.core.modeling.polyListComponentConversion(vs,fv=True,tf=True)
        fs = [pymel.core.general.PyNode(face) for face in fs]

        # Get the uv indices on the faces
        uv = []
        uv = -np.ones(np.sum(idxMap>-1),dtype='int32')
        for faces in fs:
            itr = iter(faces)
            for face in itr:
                uvs = len(face.getUVs()[0])
                uvs = [face.getUVIndex(i) for i in range(uvs)]
                vs = face.getVertices()
                vs = [idxMap[v] for v in vs]
                for v,u in zip(vs,uvs):
                    uv[v] = u
                """if any([v==-1 for v in vs]):
                    continue # Found a face with a vertex not being used
                if len(uvs) == 3:
                    uv.append(uvs)
                elif len(uvs) == 4:
                    uv.append([uvs[0],uvs[1],uvs[2]])
                    uv.append([uvs[2],uvs[3],uvs[0]])
                else:
                    logging.warning('Face with '+str(len(uvs))+' vertices encountered, ignoring face')"""
        return uv

    def getUVIndex(self,useActive=True):

        uvs = []
        if self._active is None:
            self.computeActiveVertices()
        for i in range(len(self.geoNames)):
            uv = np.asarray(self.getUVIndexOnMesh(i,useActive))
            uvs.append(uv)
        fullUV = []
        count = 0
        for u,mesh,a in zip(uvs,self.geoNodes,self._active):
            uvMesh = mesh.getUVs()
            uvMesh = [list(i) for i in uvMesh]
            uvMesh = np.asarray(uvMesh).T
            if len(uvMesh) == 0:
                continue
            fullUV.append(u+count)
            count += len(uvMesh)
        uv = np.concatenate(fullUV,0)
        return uv

    def getUVs(self):

        #if len(self.geoNames) > 1:
        #    raise NotImplementedError('Cannot get edges for multiple meshes')
        uvs = []
        for mesh in self.geoNodes:
            uv = mesh.getUVs()
            uv = [list(u) for u in uv]
            uv = np.asarray(uv).T
            uvs.append(uv)
        uvs = np.concatenate(uvs,0)
        return uvs

    def computeActiveVertices(self, reps=25):
        v0 = self.getVertices()
        eps = 1e-6
        active = np.zeros(len(v0),dtype=bool)
        for _ in range(reps):
            self.setRandomPose()
            v = self.getVertices()
            diff = np.sum(np.square(v-v0),-1)
            active = np.logical_or(active,diff>eps)
        self.active = active

    @property
    def active(self):
        return np.concatenate(self._active,0)

    @active.setter
    def active(self,value):
        verts = []
        for mesh in self.omGeoNodes:
            points = mesh.getPoints()
            verts.append(np.array(points)[:,:3])
        self._active = []
        for v,useFull in zip(verts,self.useFull):
            a = value[:len(v)].copy()
            if useFull:
                a[:] = True
            self._active.append(a)
            value = value[len(v):]

    def generateBatch(self, n=256):
        if not hasattr(self,'active'):
            self.computeActiveVertices()
        startTime = time.time()
        setTime = 0
        meshTime = 0
        mesh = np.zeros((n,np.sum(self.active),3))
        pose = np.zeros((n,self.numControls))
        for i in range(n):
            startSet = time.time()
            pose[i] = self.setRandomPose()
            endSet = time.time()
            setTime += endSet-startSet
            mesh[i] = self.getVertices()[self.active]
            endMesh = time.time()
            meshTime += endMesh-endSet
        endTime = time.time()
        logging.info('Time to generate poses: '+str(endTime-startTime)+' seconds')
        logging.info('Time to set poses: '+str(setTime)+' seconds')
        logging.info('Time to get mesh: '+str(meshTime)+' seconds')
        return pose,mesh

class PoseGeneratorRemote(PoseGenerator):

    def __init__(self,controlFile,geoFile,host,port,isServer=False):
        super(PoseGeneratorRemote,self).__init__(controlFile,geoFile)
        self.host = host
        self.port = port
        self.isServer = isServer
        self.client = None

    @property
    def active(self):
        return self._active

    def connect(self):
        # Create the connection
        client = socket.socket()
        if self.isServer:
            logging.info('Running PoseGeneratorRemote as server')
            client.bind(('',self.port))
            client.listen(1)
            connection,address = client.accept()
            self.server = client
            client = connection
        else:
            try:
                client.connect((self.host,self.port))
            except socket.error:
                logging.info('Connection to '+self.host+':'+str(self.port)+' failed')
                return
        self.client = client

    def setPose(self,pose):
        pose = np.asarray(pose).astype('float32')
        command = b'p' + packMatrix(pose.reshape(-1))
        sendMessage(command,self.client)

    def getVertices(self):
        command = b'v'
        sendMessage(command,self.client)
        msg = receiveMessage(self.client)
        return unpackMatrix(msg)

    def computeActiveVertices(self, reps=5):
        command = b'a'
        sendMessage(command,self.client)
        msg = receiveMessage(self.client)
        active = unpackMatrix(msg)
        self._active = active.astype('bool')

    def getEdges(self,useActive=True):
        if useActive:
            command = b'eA'
        else:
            command = b'e'
        sendMessage(command,self.client)
        msg = receiveMessage(self.client)
        return unpackMatrix(msg)

    def getFaces(self,useActive=True):
        if useActive:
            command = b'fA'
        else:
            command = b'f'
        sendMessage(command,self.client)
        msg = receiveMessage(self.client)
        return unpackMatrix(msg)

    def getUVs(self):
        command = b'u'
        sendMessage(command,self.client)
        msg = receiveMessage(self.client)
        return unpackMatrix(msg)

    def getUVIndex(self,useActive=True):
        if useActive:
            command = b'UA'
        else:
            command = b'U'
        sendMessage(command,self.client)
        msg = receiveMessage(self.client)
        return unpackMatrix(msg)

    def setActiveVertices(self,active):
        active = active.astype('int32')
        command = b'A'+packMatrix(active)
        sendMessage(command,self.client)
        self._active = active.astype('bool')

    def close(self):
        command = b'c'
        sendMessage(command,self.client)
        self.client.close()
        self.client = None
        if self.isServer:
            self.server.close()

def sendMessage(data,connection):
    if connection is None:
        raise RuntimeError('Connection does not exist. Is PoseGeneratorServer running?')
    if not isinstance(data,bytes):
        raise ValueError('Cannot send data of type '+str(type(data)))
    length = struct.pack('<Q',len(data))
    buff = length+data
    connection.sendall(buff)

def receiveMessage(connection):
    length = connection.recv(8)
    length = struct.unpack('<Q',length)[0]
    data = connection.recv(length)
    while len(data) != length:
        data += connection.recv(length-len(data))
    return data

def packMatrix(matrix):
    dtypes = {np.dtype('float32'):b'f',
              np.dtype('float64'):b'd',
              np.dtype('int32'):b'i'}
    dtype = ''
    for t in dtypes:
        if matrix.dtype is t:
            dtype = dtypes[t]
            break
    if dtype == '':
        raise ValueError('Cannot handle matrix dtype '+str(matrix.dtype))
    shape = matrix.shape
    data = struct.pack('c',dtype)
    data += struct.pack('B',len(shape))
    data += struct.pack('<'+'I'*len(shape),*shape)
    data += matrix.tostring()
    return data

def unpackMatrix(data):
    dtypes = {b'f':np.dtype('float32'),
              b'd':np.dtype('float64'),
              b'i':np.dtype('int32')}
    dtype = struct.unpack('c',data[0:1])[0]
    shapeLength = struct.unpack('<B',data[1:2])[0]
    shape = struct.unpack('<'+'I'*shapeLength,data[2:2+shapeLength*4])
    matrix = np.frombuffer(data[2+shapeLength*4:],dtype=dtypes[dtype]).reshape(shape)
    return matrix

class PoseGeneratorServer:

    def __init__(self,controlFile,geoFile,port=9001,isServer=True):
        if usingMaya:
            self.generator = PoseGenerator(controlFile,geoFile)
        else:
            self.generator = None
        self.port = port
        self.isServer = isServer

    def startServer(self,hostname=None):
        server = socket.socket()
        if self.isServer:
            server.bind(('',self.port))
            server.listen(1)
            connection,address = server.accept()
        else:
            logging.info('Running PoseGeneratorServer as client')
            server.connect((hostname,self.port))
            connection = server
        while True:
            buff = receiveMessage(connection)
            if not buff:
                break
            if not self.processBuffer(buff,connection):
                break
        connection.close()
        server.close()

    def processBuffer(self,buff,connection):
        commandNames = {'A':'setActiveVertices',
                        'a':'getActiveVertices',
                        'e':'getEdges',
                        'f':'getFaces',
                        'p':'setPose',
                        'u':'getUVs',
                        'U':'getUVIndex',
                        'v':'getVertices',
                        'c':'close'}
        commands = {'A':self.setActiveVertices,
                    'a':self.getActiveVertices,
                    'e':self.getEdges,
                    'f':self.getFaces,
                    'p':self.setPose,
                    'u':self.getUVs,
                    'U':self.getUVIndex,
                    'v':self.getVertices,
                    'c':self.close}
        command = buff[0:1].decode('utf-8')
        if command not in commands:
            logging.warning('Received unknown command '+str(command))
            return False
        logging.info('Received command '+commandNames[command])
        return commands[command](buff[1:],connection)

    def close(self,buff,connection):
        return False

    def setPose(self,buff,connection):
        pose = list(unpackMatrix(buff).reshape(-1))
        self.generator.setPose(pose)
        return True

    def getVertices(self,buff,connection):
        v = self.generator.getVertices().astype('float32')
        data = packMatrix(v)
        sendMessage(data,connection)
        return True

    def getEdges(self,buff,connection):
        if len(buff) > 0:
            useActive = buff[0:1].decode('utf-8')
            useActive = useActive=='A'
        else:
            useActive = False
        e = self.generator.getEdges(useActive=useActive)
        e = np.asarray(e).astype('int32')
        data = packMatrix(e)
        sendMessage(data,connection)
        return True

    def getFaces(self,buff,connection):
        if len(buff) > 0:
            useActive = buff[0:1].decode('utf-8')
            useActive = useActive=='A'
        else:
            useActive = False
        f = self.generator.getFaces(useActive=useActive)
        f = np.asarray(f).astype('int32')
        data = packMatrix(f)
        sendMessage(data,connection)
        return True

    def getUVs(self,buff,connection):
        uv = self.generator.getUVs()
        uv = np.asarray(uv).astype('float32')
        data = packMatrix(uv)
        sendMessage(data,connection)
        return True

    def getUVIndex(self,buff,connection):
        if len(buff) > 0:
            useActive = buff[0:1].decode('utf-8')
            useActive = useActive=='A'
        else:
            useActive = False
        uv = self.generator.getUVIndex(useActive=useActive)
        uv = np.asarray(uv).astype('int32')
        data = packMatrix(uv)
        sendMessage(data,connection)
        return True        

    def getActiveVertices(self,buff,connection):
        if not hasattr(self.generator,'active'):
            self.generator.computeActiveVertices()
        active = self.generator.active.astype('int32')
        data = packMatrix(active)
        sendMessage(data,connection)
        return True

    def setActiveVertices(self,buff,connection):
        active = unpackMatrix(buff).astype('bool')
        self.generator.active = active
        return True

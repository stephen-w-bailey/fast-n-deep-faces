from ctypes import *
import numpy as np
from OpenGL import GL,GLU

def computeFacesAndNormals(v, faceList):

    # Compute normals
    faces = np.asarray([v[i] for i in faceList])
    va = faces[:,0]
    vb = faces[:,1]
    vc = faces[:,2]
    diffB = vb - va
    diffC = vc - va
    vn = np.asarray([np.cross(db,dc) for db,dc in zip(diffB,diffC)])
    vn = vn / np.sqrt(np.sum(np.square(vn),-1)).reshape((-1,1))
    length = np.sqrt(np.sum(np.square(vn),-1))
    vn = np.repeat(vn.reshape((-1,1,3)),3,1)

    return faces, vn

class RenderObject(object):

    def __init__(self, v, vn, t, dynamic=False):
        self.v = v.astype('float32').reshape(-1)
        self.vn = vn.astype('float32').reshape(-1)
        self.t = t.astype('float32').reshape(-1)
        self.s = np.ones(self.v.shape[0]).astype('float32')
        if dynamic:
            self.draw = GL.GL_DYNAMIC_DRAW
        else:
            self.draw = GL.GL_STATIC_DRAW
        
        self.initialized = False
        self.visible = True

    def isInitialized(self):
        return self.initialized

    def setVisibility(self, visibility):
        self.visible = visibility

    def initializeMesh(self):

        shadow = np.ones(self.v.shape[0]).astype('float32')
        null = c_void_p(0)
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        # Vertex
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, null)
        vertices = self.v.reshape(-1)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(vertices)*4, (c_float*len(vertices))(*vertices), self.draw)

        # Normal
        self.nbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.nbo)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, null)
        normals = self.vn.reshape(-1)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(normals)*4, (c_float*len(normals))(*normals), self.draw)

        # Vertex color
        self.cbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.cbo)
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, null)
        textures = self.t.reshape(-1)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(textures)*4, (c_float*len(textures))(*textures), self.draw)

        self.line_idx = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.line_idx)
        vList = self.v.reshape((-1,3))
        n = len(vList)
        self.lineIdx = np.asarray([[i,i+1,i+1,i+2,i+2,i] for i in range(0,n-1,3)]).reshape(-1).astype('int32')
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, len(self.lineIdx)*4, (c_int*len(self.lineIdx))(*self.lineIdx), GL.GL_STATIC_DRAW)

        GL.glBindVertexArray(0)
        GL.glDisableVertexAttribArray(0)
        GL.glDisableVertexAttribArray(1)
        GL.glDisableVertexAttribArray(2)
        GL.glDisableVertexAttribArray(3)
        GL.glDisableVertexAttribArray(4)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        self.initialized = True

    def reloadMesh(self, v=None, vn=None, t=None):
        if v is not None:
            vertices = v.reshape(-1).astype('float32')
            self.v = vertices
            if self.initialized:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, len(vertices)*4, vertices, self.draw)
        if vn is not None:
            normals = vn.astype('float32').reshape(-1)
            self.vn = vn
            if self.initialized:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.nbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, len(normals)*4, normals, self.draw)
        if t is not None:
            textures = t.astype('float32').reshape(-1)
            self.t = t
            if self.initialized:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.cbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, len(textures)*4, textures, self.draw)

    def smoothNormals(self, fList):

        vn = self.vn.reshape((-1,3))
        fList = fList.reshape(-1)
        vn = np.stack([np.bincount(fList,vn[:,i]) for i in range(vn.shape[1])],1)
        vn = vn / np.sqrt(np.sum(np.square(vn),-1)).reshape((-1,1))
        vn = np.asarray([vn[f] for f in fList])

        if self.initialized:
            self.reloadMesh(vn=vn)
        else:
            self.vn = vn

    def render(self, mvp, mvpID, drawLine=False, program=-1, color=[0.0,0.0,0.0], renderDepth=False):

        if not self.initialized:
            self.initializeMesh()

        if not self.visible:
            return

        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self.v))
        GL.glBindVertexArray(0)

class InstanceObject(object):
    
    def __init__(self, renderObject):
        self.renderObject = renderObject
        self.s = None
        self.visible = True

    def setObject(self, renderObject):
        self.renderObject = renderObject

    def setOrientation(self, t, s=None):
        self.t = t
        if s is not None:
            self.s = s

    def setVisibility(self, visibility):
        self.visible = visibility

    def initialized(self):
        return self.renderObject.initialized()

    def initializeMesh(self):
        self.renderObject.initializeMesh()

    def render(self, mvp, mvpID, drawLine=False, program=-1, color=[0.0,0.0,0.0], renderDepth=False):
        if not self.visible:
            return
        if not isinstance(mvp,np.ndarray):
            mvp = np.asarray(mvp)
        mvp = mvp.reshape((4,4)).T
        t = np.eye(4)
        t[:3,3] = self.t
        if self.s is not None:
            t[:3,:3] *= self.s
        newMat = list(mvp.dot(t).T.reshape(-1).astype('float32'))
        GL.glUniformMatrix4fv(mvpID, 1, GL.GL_FALSE, (c_float*len(newMat))(*newMat))
        self.renderObject.render(mvp, mvpID, drawLine=drawLine, program=program, color=color,renderDepth=renderDepth)
        mvp = list(mvp.T.reshape(-1).astype('float32'))
        GL.glUniformMatrix4fv(mvpID, 1, GL.GL_FALSE, (c_float*len(mvp))(*mvp))

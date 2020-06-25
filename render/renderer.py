from ctypes import *
import numpy as np
from OpenGL import GL
from PySide2 import QtGui

from . import shaders

class Renderer:

    def __init__(self):
        self.idMVP = 0#GL.glGetUniformLocation(self.program, 'MVP')
        self.idAmb = 1#GL.glGetUniformLocation(self.program, 'amb')
        self.idDirect = 2#GL.glGetUniformLocation(self.program, 'direct')
        self.idDirectDir = 3#GL.glGetUniformLocation(self.program, 'directDir')
        self.idCamT = 4#GL.glGetUniformLocation(self.program, 'camT')
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        GL.glDisableClientState(GL.GL_TEXTURE_COORD_ARRAY)

    def loadProgram(self, scene):
        useProgram = shaders.program

        camera = scene.camera
        self.idMVP = GL.glGetUniformLocation(useProgram, 'MVP')
        self.idR = GL.glGetUniformLocation(useProgram, 'r')
        self.idCOP = GL.glGetUniformLocation(useProgram, 'cop')
        self.idAmb = GL.glGetUniformLocation(useProgram, 'amb')
        self.idDirect = GL.glGetUniformLocation(useProgram, 'direct')
        self.idDirectDir = GL.glGetUniformLocation(useProgram, 'directDir')
        self.idCamT = GL.glGetUniformLocation(useProgram, 'camT')
        self.idSpecAng = GL.glGetUniformLocation(useProgram, 'specAng')
        self.idSpecMag = GL.glGetUniformLocation(useProgram, 'specMag')

        self.bCoeff = [GL.glGetUniformLocation(useProgram, 'b'+str(i)) for i in range(9)]

        GL.glUseProgram(useProgram)

        null = c_void_p(0)
        
        mvp,Rinv,camT = camera.projMat()
        Rinv = list(Rinv.reshape(-1))
        mvp = mvp.astype('float32').T
        mvp = list(mvp.reshape(-1))
        amb = list(scene.amb.astype('float32'))
        direct = list(scene.direct.astype('float32'))
        directDir = list(scene.directDir.astype('float32'))
        cop = list(camera.cop.astype('float32'))

        GL.glUniform3fv(self.idAmb, 1, (c_float*len(amb))(*amb))
        GL.glUniform3fv(self.idDirect, 1, (c_float*len(direct))(*direct))
        GL.glUniform3fv(self.idDirectDir, 1, (c_float*len(directDir))(*directDir))
        GL.glUniform3fv(self.idCamT, 1, (c_float*len(camT))(*camT))
        GL.glUniform2fv(self.idCOP, 1, (c_float*len(cop))(*cop))
        GL.glUniformMatrix4fv(self.idMVP, 1, GL.GL_FALSE, (c_float*len(mvp))(*mvp))
        GL.glUniformMatrix3fv(self.idR, 1, GL.GL_FALSE, (c_float*len(Rinv))(*Rinv))
        GL.glUniform1f(self.idSpecAng,5)
        GL.glUniform1f(self.idSpecMag,0.1)

        return mvp,useProgram

    def renderScene(self,scene,clear=True):
        if clear:
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        mvp,program = self.loadProgram(scene)
        for obj in scene.meshes:
            obj.render(mvp,self.idMVP,program=program)

        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

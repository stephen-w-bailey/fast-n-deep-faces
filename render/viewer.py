from PySide2 import QtGui, QtWidgets, QtCore
from OpenGL import GL
import matplotlib.pyplot as plt
import numpy as np

from . import renderer
from . import shaders

class Viewer(QtWidgets.QOpenGLWidget):

    clicked = QtCore.Signal(int,int)
    dragged = QtCore.Signal(int,int)

    def __init__(self,scene,parent=None):
        QtWidgets.QOpenGLWidget.__init__(self,parent)
        self.scene = scene
        self.clickRotate = True
        self.clickZoom = True

    def setMouse(self,rotate=True,zoom=True):
        self.clickRotate = rotate
        self.clickZoom = zoom

    def initializeGL(self):
        print('Initializing gl')
        self.context().functions().glClearColor(1.0,1.0,1.0,1.0)
        shaders.initialize()
        self.renderer = renderer.Renderer()

    def resizeGL(self, width, height):
        self.imgWidth = width
        self.imgHeight = height
        side = width
        self.side = side
        GL.glViewport(int((width - side) / 2),int((height - side) / 2), side, side)
        self.scene.camera.setAspect(height/width)

    def deproject(self,x,y):
        x = (x/self.imgWidth)*2-1
        y = (y/self.imgHeight)*2-1
        y = y*self.imgHeight/self.imgWidth
        screen = np.asarray([x,-y])
        return self.scene.camera.getRay(screen)

    def minimumSizeHint(self):
        return QtCore.QSize(600, 400)

    def paintGL(self):
        self.renderer.renderScene(self.scene)

    def mousePressEvent(self,event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.clicked.emit(event.x(),event.y())
        if self.clickRotate or self.clickZoom:
            point = np.asarray([event.x(),event.y()])
            self.scene.camera.down(point)

    def mouseMoveEvent(self,event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.dragged.emit(event.x(),event.y())
        point = np.asarray([event.x(),event.y()])
        if self.clickRotate:
            if event.buttons() & QtCore.Qt.LeftButton:
                self.scene.camera.drag(point)
        if self.clickZoom:
            if event.buttons() & QtCore.Qt.RightButton:
                self.scene.camera.zoom(point)
        self.update()

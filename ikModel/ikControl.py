import argparse
import cv2
from moviepy.editor import ImageSequenceClip
import numpy as np
from PySide2 import QtCore,QtGui,QtWidgets
import os
import pickle
from sklearn.decomposition import PCA
import sys
import tensorflow as tf
import trimesh
import yaml
from pathlib import Path

basePath = (Path(__file__).parent / '..').resolve()
sys.path.append(os.path.join(basePath, 'cnnModel'))
sys.path.append(os.path.join(basePath, '.'))

import cnnOpt
import ikModel
import ikOpt
import render
import rigidDeformer

def identifyBorderPoints(faces,vMask):
    points = np.arange(np.max(faces)+1)
    inPoints = points[vMask]
    boundary = []
    for f in faces:
        for e1,e2 in zip(f,[f[1],f[2],f[0]]):
            if e1 in inPoints and e2 not in inPoints:
                boundary.append(e1)
            elif e2 in inPoints and e1 not in inPoints:
                boundary.append(e2)
    return boundary

class Viewer(QtWidgets.QWidget):

    updateMesh = QtCore.Signal(np.ndarray,np.ndarray,np.ndarray)

    def __init__(self,config,checkpoint,sess,meshConfig=None,meshCheckpoint=None):
        super(Viewer,self).__init__()
        self.captureData = None
        self.neutralPose = None
        self.currentPoints = None
        self.sess = sess
        self.buildModel(config,checkpoint,meshConfig,meshCheckpoint)
        self.buildGUI(config)
        
    def buildModel(self,config,checkpoint,meshConfig=None,meshCheckpoint=None):

        # Build the model graph
        if meshConfig is None:
            with open(os.path.join(basePath,config['training_params']['approximation_config'])) as file:
                approximationConfig = yaml.load(file)
        else:
            with open(meshConfig) as file:
                approximationConfig = yaml.load(file)
        with open(os.path.join(basePath,config['data_params']['point_config']),'rb') as file:
            pointData = pickle.load(file)
        inputShape = (1,len(pointData['faces']),2)
        self.points = tf.placeholder(tf.float32,inputShape)
        data = dict(points=self.points)
        varsBefore = tf.trainable_variables()
        self.model = ikModel.buildModel(data,config)
        self.rigControls = self.model['output']
        varsAfter = tf.trainable_variables()
        if 'base_config' in approximationConfig['model_params']:
            with tf.variable_scope('refine'):
                self.refineMesh = ikOpt.buildModel(approximationConfig,self.model['output'],addNeutral=False)
                mask = self.refineMesh['cache']['vCharts']>=0
                boundary = identifyBorderPoints(self.refineMesh['cache']['faces'],mask)
            with open(os.path.join(basePath,approximationConfig['model_params']['base_config'])) as file:
                baseConfig = yaml.load(file)
            self.mesh = ikOpt.buildModel(baseConfig,self.model['output'])
            self.refineMesh['output'] = self.mesh['output'] + self.refineMesh['output']
            approximationConfig = baseConfig
        else:
            self.mesh = ikOpt.buildModel(approximationConfig,self.model['output'])
        varsAfterMesh = tf.trainable_variables()

        # Load info about the mesh
        with open(os.path.join(basePath,approximationConfig['data_params']['cache_file']),'rb') as file:
            data = pickle.load(file)
        parts = data['vCharts']
        neutral = data['neutral'][data['active']]
        faces = data['faces']
        mask = np.arange(len(parts))[parts>-1]
        pointFaces = faces[pointData['faces']]
        bary = pointData['bary']
        points = neutral[pointFaces.reshape(-1)].reshape((-1,3,3))
        points = np.sum(points*bary[...,np.newaxis],1)
        points[...,2] = -50
        self.neutralPoints = points[...,:2]

        # Apply ridig deformer
        if 'rigid_files' in approximationConfig['data_params']:
            self.rigidDeformer = rigidDeformer.RigidDeformer(neutral,[os.path.join(basePath,f) for f in approximationConfig['data_params']['rigid_files']],mask)
            self.mesh = self.rigidDeformer.deformTF(self.mesh['output'][0])[np.newaxis]
            if hasattr(self,'refineMesh'):
                self.refineMesh = self.rigidDeformer.deformTF(self.refineMesh['output'][0])[np.newaxis]
        else:
            self.mesh = self.mesh['output']
            if hasattr(self,'refineMesh'):
                self.refineMesh = self.refineMesh['output']

        # Compute the normals
        v = self.mesh[0]
        normals = cnnOpt.faceNormals(v,faces)
        self.normals = cnnOpt.vertexNormals(v,normals,faces)
        if hasattr(self,'refineMesh'):
            v = self.refineMesh[0]
            normals = cnnOpt.faceNormals(v,faces)
            self.refineNormals = cnnOpt.vertexNormals(v,normals,faces)

        # Load the model from file
        varList = [v for v in varsAfter if v not in varsBefore]
        saver = tf.train.Saver(varList)
        checkpointFile = tf.train.latest_checkpoint(checkpoint)
        saver.restore(self.sess,checkpointFile)
        varList = [v for v in varsAfterMesh if v not in varsAfter]
        saver = tf.train.Saver(varList)
        checkpointFile = tf.train.latest_checkpoint(checkpoint if meshCheckpoint is None else meshCheckpoint)
        saver.restore(self.sess,checkpointFile)

    def buildGUI(self,config):
        with open(os.path.join(basePath,config['training_params']['approximation_config'])) as file:
            approximationConfig = yaml.load(file)
        with open(os.path.join(basePath,approximationConfig['data_params']['cache_file']),'rb') as file:
            data = pickle.load(file)
        faces = data['faces']
        self.faces = faces
        neutral = data['neutral'][data['active']]
        self.mean = np.mean(neutral,0)
        neutral = neutral-self.mean
        mesh = trimesh.Trimesh(neutral,faces,process=False)
        vertices = neutral[faces.reshape(-1)].reshape((-1,3,3))
        normals = np.repeat(mesh.face_normals[:,np.newaxis],3,1)
        texture = 0.75 * np.ones(normals.shape)
    
        # Load the mesh viewer
        obj = render.mesh.RenderObject(vertices,normals,texture,dynamic=True)
        scene = render.Scene()
        scene.addMesh(obj)
        self.view = render.Viewer(scene)
        self.view.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,QtWidgets.QSizePolicy.MinimumExpanding)
        self.meshObject = obj

        # Add some toggles
        self.refineCheck = QtWidgets.QCheckBox('Apply refinement')
        self.refineCheck.setChecked(True)
        self.refineCheck.stateChanged.connect(self.onRefineClick)

        # Put the gui together
        gridLayout = QtWidgets.QGridLayout(self)
        gridLayout.addWidget(self.view,0,0,1,1)
        gridLayout.addWidget(self.refineCheck,1,0,1,1)

    def onRefineClick(self):
        self.setFrame(self.currentPoints)

    def setFrame(self,points):

        # Compute the new points
        self.currentPoints = points

        # Determine which mesh to compute
        if self.refineCheck.isChecked():
            mesh = self.refineMesh
            normals = self.refineNormals
        else:
            mesh = self.mesh
            normals = self.normals

        # Run the points through the model
        feed_dict = {self.points:points[np.newaxis]}
        newMesh,newNormals,params = self.sess.run((mesh,normals,self.rigControls),feed_dict=feed_dict)
        newMesh = newMesh[0]-self.mean
        vertices = newMesh[self.faces.reshape(-1)].reshape((-1,3,3))
        newNormals = newNormals[self.faces.reshape(-1)].reshape((-1,3,3))
        self.view.makeCurrent()
        self.meshObject.reloadMesh(vertices,newNormals)
        self.view.update()
        self.updateMesh.emit(vertices,newNormals,params)

class Controller(QtWidgets.QWidget):

    updatePoints = QtCore.Signal(np.ndarray)

    def __init__(self,config):
        super(Controller,self).__init__()
                      
        # Load info about the mesh
        with open(os.path.join(basePath,config['training_params']['approximation_config'])) as file:
            approximationConfig = yaml.load(file)
        with open(os.path.join(basePath,approximationConfig['data_params']['cache_file']),'rb') as file:
            data = pickle.load(file)
        with open(os.path.join(basePath,config['data_params']['point_config']),'rb') as file:
            self.pointData = pickle.load(file)

        # Load the mesh
        self.neutral = data['neutral'][data['active']]
        self.neutralMean = np.mean(self.neutral,0)
        self.neutral = self.neutral-self.neutralMean
        self.faces = data['faces']
        self.savedPoints = []
        self.savedParams = []
        mesh = trimesh.Trimesh(self.neutral,self.faces,process=False)
        vertices = self.neutral[self.faces.reshape(-1)].reshape((-1,3,3))
        normals = np.repeat(mesh.face_normals[:,np.newaxis],3,1)
        texture = 0.75 * np.ones(normals.shape)

        # Load the sphere
        sphere = []
        sphereF = []
        with open(os.path.join(basePath,'data','sphere.obj')) as file:
            for line in file:
                parts = line.strip().split(' ')
                if parts[0] == 'v':
                    sphere.append([float(i) for i in parts[1:4]])
                elif parts[0] == 'f':
                    f = []
                    for segment in parts[1:4]:
                        f.append(int(segment.split('/')[0])-1)
                    sphereF.append(f)
        sphere = np.asarray(sphere)*0.25
        sphereF = np.asarray(sphereF)
        sphereV = sphere[sphereF.reshape(-1)].reshape((-1,3,3))
        mesh = trimesh.Trimesh(sphere,sphereF,process=False)
        sphereN = np.repeat(mesh.face_normals[:,np.newaxis],3,1)
        sphereT = np.zeros(normals.shape)
        sphereT[...,1] = 1
        self.greenSphere = render.mesh.RenderObject(sphereV,sphereN,sphereT,dynamic=False)
        sphereT = np.zeros(normals.shape)
        sphereT[...,0] = 1
        self.redSphere = render.mesh.RenderObject(sphereV,sphereN,sphereT,dynamic=False)

        # Load the mesh viewer
        obj = render.mesh.RenderObject(vertices,normals,texture,dynamic=True)
        scene = render.Scene()
        scene.addMesh(obj)
        self.view = render.Viewer(scene)
        self.view.setMouse(False)
        scene.camera.setProjection('orthographic')
        self.view.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,QtWidgets.QSizePolicy.MinimumExpanding)
        self.meshObject = obj
        self.view.clicked.connect(self.onClicked)
        self.view.dragged.connect(self.onDragged)
        self.meshObj = obj
        self.curV = vertices

        # Position the spheres
        pointFaces = self.faces[self.pointData['faces']]
        bary = self.pointData['bary']
        points = self.neutral[pointFaces.reshape(-1)].reshape((-1,3,3))
        points[...,2] = 50
        self.points = np.sum(points*bary[...,np.newaxis],1)
        self.spheres = []
        for p in self.points:
            p = p.copy()
            p[-1] = 20
            s = render.mesh.InstanceObject(self.redSphere)
            s.setOrientation(p,0.2)
            s.setVisibility(True)
            self.spheres.append(s)
            scene.addMesh(s)
        self.selectedPoint = None

        # Add recording functions
        self.recordButton = QtWidgets.QPushButton('Record')
        self.playButton = QtWidgets.QPushButton('Play')
        self.saveButton = QtWidgets.QPushButton('Save Pose')
        self.isRecording = False
        self.recordButton.clicked.connect(self.onRecord)
        self.playButton.clicked.connect(self.onPlay)
        self.saveButton.clicked.connect(self.onSave)

        # Put the gui together
        gridLayout = QtWidgets.QGridLayout(self)
        gridLayout.addWidget(self.view,0,0,1,2)
        gridLayout.addWidget(self.recordButton,1,0,1,1)
        gridLayout.addWidget(self.playButton,1,1,1,1)
        gridLayout.addWidget(self.saveButton,2,0,1,2)

    def onUpdateMesh(self,v,n,p):
        self.view.makeCurrent()
        self.meshObj.reloadMesh(v,n)
        self.curV = v
        self.curParams = p

    def onRecord(self):
        if self.isRecording:
            self.stopRecording()
        else:
            self.startRecording()
        self.isRecording = not self.isRecording

    def onSave(self):
        self.savedPoints.append(self.points.copy())
        self.savedParams.append(self.curParams)
        print('Saved '+str(len(self.savedPoints))+' poses')

    def startRecording(self):
        self.recordButton.setText('Stop')
        self.recording = []
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.recordFrame)
        self.timer.start(33)

    def stopRecording(self):
        self.recordButton.setText('Record')
        self.timer.stop()
        fileName = QtWidgets.QFileDialog.getSaveFileName(caption='Save Recording',filter='*.pkl')
        print('Saving to '+str(fileName[0]))
        if fileName[0] != '':
            with open(fileName[0],'wb') as file:
                pickle.dump(self.recording,file)

    def recordFrame(self):
        self.recording.append((self.points.copy(),self.selectedPoint))

    def onPlay(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(caption='Load Recording',filter='*.pkl')
        if fileName[0] == '':
            return
        with open(fileName[0],'rb') as file:
            self.recording = pickle.load(file)
        self.frame = 0
        self.images = []
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.onFrame)
        self.timer.start(33)

    def onFrame(self):
        frame = self.recording[self.frame]
        points,selectedPoint = frame

        # Update the selected point
        if self.selectedPoint != selectedPoint:
            if self.selectedPoint is not None:
                self.spheres[self.selectedPoint].setObject(self.redSphere)
            if selectedPoint is not None:
                self.spheres[selectedPoint].setObject(self.greenSphere)
            self.selectedPoint = selectedPoint

        # Update the point positions
        for p,s in zip(points,self.spheres):
            s.setOrientation(p)

        # Update the screen
        self.points = points
        updatePoints = self.points[:,:2] + self.neutralMean[:2]
        self.updatePoints.emit(updatePoints)
        self.view.update()

        # Save the screen
        image = self.view.grabFramebuffer()
        image.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        width = image.width()
        height = image.height()
        ptr = image.constBits()
        arr = np.array(ptr).reshape(height, width, 4)
        arr = arr[...,[2,1,0]]
        self.images.append(arr)
        
        self.frame += 1
        if self.frame >= len(self.recording):
            self.stopPlay()

    def stopPlay(self):
        self.timer.stop()
        fileName = QtWidgets.QFileDialog.getSaveFileName(caption='Save Recording',filter='*.mp4')
        print('Saving to '+str(fileName[0]))
        if fileName[0] != '':
            clip = ImageSequenceClip(self.images,fps=30)
            clip.write_videofile(fileName[0],fps=30)

    def onClicked(self,x,y):

        # Cet the closest point
        p,_ = self.view.deproject(x,y)
        points = self.curV[self.pointData['faces']]
        points = np.sum(points*self.pointData['bary'][...,np.newaxis],1)
        d = np.sum(np.square(points[:,:2]-p[:2]),-1)
        idx = np.argmin(d)

        # Update the selected sphere
        if self.selectedPoint is not None:
            self.spheres[self.selectedPoint].setObject(self.redSphere)
        self.spheres[idx].setObject(self.greenSphere)
        self.selectedPoint = idx
        self.offset = p[:2]
        self.startPoint = self.points[idx,:2].copy()

        self.view.update()

    def onDragged(self,x,y):

        # Update the point
        p,_ = self.view.deproject(x,y)
        delta = p[:2]-self.offset
        newPoint = self.startPoint+delta
        self.points[self.selectedPoint,:2] = newPoint
        self.spheres[self.selectedPoint].setOrientation(self.points[self.selectedPoint])

        # Update the mesh
        updatePoints = self.points[:,:2] + self.neutralMean[:2]
        self.updatePoints.emit(updatePoints)

        # Render
        self.view.update()

    def closeEvent(self,event):
        if len(self.savedPoints) > 0:
            fileName = QtWidgets.QFileDialog.getSaveFileName(caption='Save Poses',filter='*.pkl')
            if fileName[0] != '':
                with open(fileName[0],'wb') as file:
                    res = dict(points=np.asarray(self.savedPoints)[:,0],
                               pose=np.asarray(self.savedParams)[:,0])
                    pickle.dump(res,file)
                
        return super().closeEvent(event)

def main():
    parser = argparse.ArgumentParser(description='View performance capture model')
    parser.add_argument('--configFile', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--meshConfig', type=str)
    parser.add_argument('--meshCheckpoint', type=str)
    parser.add_argument('--camPos', type=float)
    args = parser.parse_args()
    with open(args.configFile) as file:
        config = yaml.load(file)

    with tf.Session() as sess:
        app = QtWidgets.QApplication(sys.argv)
        v = Viewer(config,args.checkpoint,sess,meshConfig=args.meshConfig,meshCheckpoint=args.meshCheckpoint)
        c = Controller(config)
        c.updatePoints.connect(v.setFrame)
        v.updateMesh.connect(c.onUpdateMesh)
        v.currentPoints = c.points[:,:2]+c.neutralMean[:2]
        if args.camPos is not None:
            c.view.scene.camera.pose[3,2] = args.camPos
        c.resize(1080,1180)
        v.show()
        c.show()

        sys.exit(app.exec_())

if __name__=='__main__':
    main()

import argparse
import functools
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from skimage.transform import resize
import sys
import tensorflow as tf
import time
import trimesh
import yaml
from pathlib import Path

basePath = (Path(__file__).parent).resolve()
sys.path.append(os.path.join(basePath, 'cnnModel'))

from PySide2 import QtCore,QtWidgets

import cnnModel
import cnnOpt
import render
import UVGenerator
import rigidDeformer

class Viewer(QtWidgets.QWidget):

    def __init__(self,rigData,config,animFile,checkpoint,sess):
        super(Viewer,self).__init__()
        self.loadCache(rigData,config)

        if 'base_config' in config['model_params']:
            print('Loading refinement')
            with tf.variable_scope('refine'):
                self.buildModel(config,animFile,applyNeutral=False)
            self.refineModel = self.model
            print('Loading base model')
            with open(os.path.join(basePath,config['model_params']['base_config'])) as file:
                baseConfig = yaml.load(file)
            print('Loading cache file '+baseConfig['data_params']['cache_file'])
            with open(os.path.join(basePath,baseConfig['data_params']['cache_file']),'rb') as file:
                baseCache = pickle.load(file)
            self.loadCache(baseCache,baseConfig)
            self.buildModel(baseConfig,animFile)
            self.model['refine_output'] = self.model['output'] + self.refineModel['output']
        else:
            self.buildModel(config,animFile)
            baseConfig = config
        #self.images = self.model['images']
            
        # Load the mesh viewer
        obj = render.mesh.RenderObject(self.vertices,self.normals,self.texture,dynamic=True)
        scene = render.Scene()
        scene.addMesh(obj)
        self.view = render.Viewer(scene)
        self.view.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,QtWidgets.QSizePolicy.MinimumExpanding)
        self.mesh = obj

        # Load the control list
        controlFile = config['data_params']['control_file']
        controls = []
        controlDefault = []
        controlRange = []
        with open(controlFile) as file:
            for line in file:
                if ',' in line:
                    parts = line.strip().split(',')
                else:
                    parts = line.strip().split(' ')
                    parts = [parts[0]+'-'+parts[1]]+parts[2:]
                controls.append(parts[0]+'-'+parts[1])
                controlRange.append((float(parts[2]),float(parts[3])))
                if len(parts) > 4:
                    controlDefault.append(float(parts[4]))
                else:
                    controlDefault.append(0.0)
        self.controlRange = controlRange

        # Load the timeline
        if self.anim is not None:
            timeline = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            timeline.setRange(0,len(self.anim)-1)
            timeline.valueChanged.connect(self.onTimelineMove)
            self.timelineText = QtWidgets.QLabel('1/'+str(len(self.anim)))
            self.timeline = timeline
            self.playButton = QtWidgets.QPushButton('Play')
            self.playTimer = QtCore.QTimer()
            #self.playTimer.setInterval(33)
            self.playButton.clicked.connect(self.onPlayPress)
            self.playTimer.timeout.connect(self.onNextFrame)

        # Create the control GUI
        controlWidget = QtWidgets.QWidget()
        controlLayout = QtWidgets.QGridLayout(controlWidget)
        controlWidget.setSizePolicy(QtWidgets.QSizePolicy.Maximum,QtWidgets.QSizePolicy.Minimum)
        controlList = QtWidgets.QListWidget()
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollArea.setWidget(controlWidget)
        scrollArea.setSizePolicy(QtWidgets.QSizePolicy.Maximum,QtWidgets.QSizePolicy.MinimumExpanding)
        self.refineCheck = QtWidgets.QCheckBox('Apply refinement')
        self.refineCheck.setChecked(True)
        self.refineCheck.stateChanged.connect(self.onRefineClick)

        # Create the animation selection widget
        if animFile is not None:
            self.animSelect = QtWidgets.QListWidget()
            for k in self.animNames:
                length = len(self.fullAnim[k])
                label = k+' ('+str(length)+')'
                self.animSelect.addItem(label)
            self.animSelect.show()
            self.animSelect.currentRowChanged.connect(self.onAnimChanged)

        self.sliders = []
        self.sliderText = []
        for i in range(len(controls)):
                label = QtWidgets.QLabel(controls[i])
                value = QtWidgets.QLabel('{0:.3f}'.format(controlDefault[i]))
                slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                slider.setRange(0,1000)
                slider.setMinimumWidth(300)
                controlLayout.addWidget(label,i,0,1,1)
                controlLayout.addWidget(value,i,1,1,1)
                controlLayout.addWidget(slider,i,2,1,1)
                sliderFunc = functools.partial(self.onSliderChange,index=i)
                slider.valueChanged.connect(sliderFunc)
                slider.sliderMoved.connect(self.onSliderMoved)
                self.sliders.append(slider)
                self.sliderText.append(value)
                
        # Put the gui together
        gridLayout = QtWidgets.QGridLayout(self)
        gridLayout.addWidget(self.view,0,0,1,1)
        gridLayout.addWidget(scrollArea,0,1,1,1)
        if animFile is not None:
            gridLayout.addWidget(timeline,1,0,1,2)
            gridLayout.addWidget(self.timelineText,2,0,1,1)
            gridLayout.addWidget(self.playButton,2,1,1,1)
        gridLayout.addWidget(self.refineCheck,3,1,1,1)

        self.status = QtWidgets.QStatusBar()
        gridLayout.addWidget(self.status,3,0,1,2)

        self.sess=sess
        if os.path.isdir(checkpoint):
            checkpoint = tf.train.latest_checkpoint(checkpoint)
        else:
            checkpoint = checkpoint
        saver = tf.train.Saver()
        saver.restore(sess,checkpoint)

        # Compute rigid parts and normals
        with tf.device('/CPU:0'):
        
            self.rigidDeformer = None
            if 'rigid_files' in baseConfig['data_params']:
                mask = np.arange(len(self.neutral))[self.parts>-1]
                self.rigidDeformer = rigidDeformer.RigidDeformer(self.neutral,baseConfig['data_params']['rigid_files'],mask)
                self.model['output'] = self.rigidDeformer.deformTF(self.model['output'][0])[np.newaxis]

            v = self.model['output'][0]
            normals = cnnOpt.faceNormals(v,self.faces)
            normals = cnnOpt.vertexNormals(v,normals,self.faces)
            normals = tf.gather(normals,self.faces.reshape(-1))
            normals = tf.reshape(normals,(-1,3,3))
            #normals = tf.tile(normals[:,np.newaxis],(1,3,1))
            self.model['normals'] = normals
            if 'refine_output' in self.model:
                if 'rigid_files' in baseConfig['data_params']:
                    self.model['refine_output'] = self.rigidDeformer.deformTF(self.model['refine_output'][0])[np.newaxis]
                v = self.model['refine_output'][0]
                normals = cnnOpt.faceNormals(v,self.faces)
                normals = cnnOpt.vertexNormals(v,normals,self.faces)
                normals = tf.gather(normals,self.faces.reshape(-1))
                normals = tf.reshape(normals,(-1,3,3))
                #normals = tf.tile(normals[:,np.newaxis],(1,3,1))
                self.model['refine_normals'] = normals

        if animFile is not None:
            self.setFrame(0)
        else:
            self.setPose(controlDefault)

    def onSliderChange(self,value,index):
        r = self.controlRange[index]
        text = self.sliderText[index]
        v = value/1000
        v = v*(r[1]-r[0])+r[0]
        text.setText('{0:.3f}'.format(v))

    def onSliderMoved(self,value):
        pose = self.getSliderPose()
        self.setPose(pose)

    def onAnimChanged(self,row):
        k = self.animNames[row]
        self.anim = self.fullAnim[k]
        self.setFrame(0)
        self.timeline.setSliderPosition(0)
        self.timeline.setRange(0,len(self.anim)-1)
        self.status.showMessage('Showing '+k)

    def loadCache(self,rigData,config):
        
        # Load the mesh data
        self.data = rigData
        neutral = self.data['neutral']/1.5
        active = self.data['active']
        neutral = neutral[active].astype('float32')
        meshMean = np.mean(neutral,0)
        neutral = neutral - meshMean
        faces = self.data['faces']
        parts = self.data['vCharts']
        uv = self.data['uv']
        mesh = trimesh.Trimesh(neutral,faces,process=False)
        vertices = neutral[faces.reshape(-1)].reshape((-1,3,3))
        normals = np.repeat(mesh.face_normals[:,np.newaxis],3,1)
        texture = 0.75 * np.ones(normals.shape)

        if 'hide_unused' in config['model_params'] and config['model_params']['hide_unused']:
            unusedVerts = parts==-1
            faceCount = unusedVerts[faces.reshape(-1)].reshape((-1,3))
            faceCount = np.sum(faceCount,-1)
            faces = faces[faceCount==0]
        
        self.faces = faces
        self.parts = parts
        self.neutral = neutral
        self.uv =  uv
        if 'parameter_mask' in self.data:
            self.mask = self.data['parameter_mask']
        else:
            self.mask = None
        self.vertices = vertices
        self.normals = normals
        self.texture = texture

    def buildModel(self,config,animFile,applyNeutral=True):

        # Load the animation sequence
        if animFile is not None:
            with open(animFile,'rb') as file:
                anim = pickle.load(file,encoding='latin1')
            self.fullAnim = anim
            self.animNames = sorted([k for k in anim])
            self.anim = anim[self.animNames[0]]
            controls = self.anim.shape[1]
        else:
            self.anim = None
            controls = 0
            with open(config['data_params']['control_file']) as file:
                for _ in file:
                    controls += 1

        # Create the model
        if not hasattr(self,'posePH'):
            self.posePH = tf.placeholder(tf.float32,shape=(1,controls))
        partCount = np.max(self.parts)+1
        data = {'pose':self.posePH}
        self.usedVerts = []
        self.usedUVs = []
        for i in range(partCount):
            if np.sum(self.parts==i) > 0:
                data['image-'+str(i)] = tf.ones(1)
            else:
                data['image-'+str(i)] = None
            ref = self.faces.reshape(-1)
            idx = np.arange(len(self.neutral))[self.parts==i]
            if len(idx) == 0:
                continue
            uv = self.uv[idx]
            self.usedUVs.append(uv)
            self.usedVerts.append(idx)
        idx = np.concatenate(self.usedVerts)
        linear = np.zeros(self.neutral.shape,dtype='float32')
        if applyNeutral:
            linear[idx] = self.neutral[idx]
            neutral = self.neutral
        else:
            neutral = linear
        data['linear'] = linear
        self.model = cnnModel.buildModel(data,self,neutral,config)

    def setSelection(self,index):
        texture = 0.75 * np.ones(self.neutral.shape,dtype='float32')
        texture[self.parts==index] = [1,0,0]
        texture = texture[self.faces.reshape(-1)].reshape((-1,3,3))
        self.mesh.reloadMesh(t=texture)

    def getUV(self,index):
        usedV = np.zeros(len(self.neutral),dtype='bool')
        usedV[self.parts == index] = True
        usedT = usedV[self.faces.reshape(-1)].reshape((-1,3))
        usedT = np.sum(usedT,-1)==3
        usedVIdx = np.arange(len(self.neutral))[usedV]
        usedTIdx = np.asarray(list(set(list(self.faces[usedT].reshape(-1)))))
        diff = len(usedVIdx)-len(usedTIdx)
        print('Vertices missing in triangles: '+str(diff))
        plt.triplot(self.uv[:,0],self.uv[:,1],self.faces[usedT])
        plt.axis('equal')
        plt.show()

    def onNextFrame(self):
        frame = (self.timeline.value()+1)
        if frame < len(self.anim):
            self.timeline.setSliderPosition(frame)
        else:
            self.stop()

    def onTimelineMove(self,value):
        self.timelineText.setText(str(value+1)+'/'+str(len(self.anim)))
        self.setFrame(value)

    def setFrame(self,index):
        pose = self.anim[index]
        self.setPose(pose)

    def onPlayPress(self):
        if self.playTimer.isActive():
            self.stop()
        else:
            self.play()

    def stop(self):
        self.playButton.setText('Play')
        self.playTimer.stop()

    def play(self):
        self.playButton.setText('Stop')
        self.playTimer.start(33)

    def getSliderPose(self):
        pose = []
        for r,slider in zip(self.controlRange,self.sliders):
            v = slider.sliderPosition()/1000
            v = v*(r[1]-r[0])+r[0]
            pose.append(v)
        pose = np.asarray(pose).astype('float32')
        return pose

    def onRefineClick(self):
        pose = self.getSliderPose()
        self.setPose(pose)

    def setPose(self,pose):
        for i in range(len(pose)):
            v = pose[i]
            r = self.controlRange[i]
            v = int((v-r[0])/(max(r[1]-r[0],1e-6))*1000)
            self.sliders[i].setSliderPosition(v)

        pose = self.getSliderPose()

        if self.refineCheck.isChecked() and 'refine_output' in self.model:
            output = 'refine_output'
            normals = 'refine_normals'
        else:
            output = 'output'
            normals = 'normals'

        feed_dict = {self.posePH:pose[np.newaxis]}
        start = time.time()
        verts,normals = self.sess.run((self.model[output],self.model[normals]),feed_dict=feed_dict)
        verts = verts[0]
        elapsed = time.time()-start
        #if self.rigidDeformer is not None:
        #    verts = self.rigidDeformer.deform(verts)
        #mesh = trimesh.Trimesh(verts,self.faces,process=False)
        vertices = verts[self.faces.reshape(-1)].reshape((-1,3,3))
        #normals = np.repeat(mesh.face_normals[:,np.newaxis],3,1)
        self.mesh.reloadMesh(vertices,normals)
        self.view.update()
        self.status.showMessage('Evaluation time: {:.0f}ms'.format(elapsed*1000))
        
        
def main():
    parser = argparse.ArgumentParser(description='Visualize a mesh for parameterization')
    parser.add_argument('--configFile', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--animFile', type=str)
    args = parser.parse_args()
    with open(args.configFile) as file:
        config = yaml.load(file)

    with open(os.path.join(basePath,config['data_params']['cache_file']),'rb') as file:
        data = pickle.load(file)
    neutral = data['neutral']
    active = data['active']
    neutral = neutral[active].astype('float32')
    meshMean = np.mean(neutral,0)
    neutral = neutral - meshMean
    faces = data['faces']
    parts = data['vCharts']
    #parts = 15*np.ones(len(faces),dtype='int32')

    mesh = trimesh.Trimesh(neutral,faces,process=False)
    vertices = neutral[faces.reshape(-1)].reshape((-1,3,3))
    normals = np.repeat(mesh.face_normals[:,np.newaxis],3,1)
    texture = 0.75 * np.ones(normals.shape)

    with tf.Session() as sess:

        app = QtWidgets.QApplication(sys.argv)
        v = Viewer(data,config,args.animFile,args.checkpoint,sess)
        v.show()

        partIndex = -1
        maxParts = np.max(parts)+1
        visibleParts = np.zeros(maxParts,dtype=bool)

        def nextSample():
            nonlocal partIndex
            partIndex = (partIndex+1)%maxParts
            print('Showing part '+str(partIndex))
            texture = 0.75 * np.ones(normals.shape,dtype='float32')
            texture[parts==partIndex] = [1,0,0]
            mesh.reloadMesh(t=texture)
            v.update()

        def onKeyboard(key):
            print('Pressed '+str(key))
            if key == render.viewer.pygame.K_v:
                visibleParts[partIndex] = True
                nextSample()
            if key == render.viewer.pygame.K_SPACE:
                nextSample()
        #v.addKeyboardCallback(onKeyboard)

        #nextSample()
        sys.exit(app.exec_())
    
if __name__=='__main__':
    main()

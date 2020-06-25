import argparse
import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle

import PoseGenerator

# Computes parts by connected components in the uv space
def computeParts(uv,faces):

    # Remove triangles with long edges
    threshold = 0.25
    uvTri = uv[faces.reshape(-1)].reshape((-1,3,2))
    a,b,c = uvTri[:,1]-uvTri[:,0],uvTri[:,2]-uvTri[:,0],uvTri[:,2]-uvTri[:,1]
    uvTri = np.stack((a,b,c),axis=1)
    lengths = np.sqrt(np.sum(np.square(uvTri),-1))
    maxLength = np.max(lengths,-1)
    faces = faces[maxLength<threshold]
    
    g = nx.Graph()
    for f in faces:
        g.add_edge(f[0],f[1])
        g.add_edge(f[1],f[2])
        g.add_edge(f[2],f[0])
    components = [i for i in nx.connected_components(g)]
    parts = -np.ones(len(uv),dtype='int32')
    for idx,c in enumerate(components):
        for i in c:
            parts[i] =idx

    return parts

def main():
    parser = argparse.ArgumentParser(description='Create cache file')
    parser.add_argument('--refCache', type=str)
    parser.add_argument('--cacheFile', type=str, required=True)
    parser.add_argument('--controlFile', type=str, required=True)
    parser.add_argument('--geoFile', type=str, required=True)
    parser.add_argument('--activeVertices', type=str)
    args = parser.parse_args()
    
    pg=PoseGenerator.PoseGeneratorRemote(args.controlFile,args.geoFile,'localhost',9001)
    pg.connect()
    try:
        if args.refCache is not None:
            with open(args.refCache,'rb') as file:
                refCache = pickle.load(file)
            pg.setActiveVertices(refCache['active'])
        else:
            if args.activeVertices is not None:
                with open(args.activeVertices,'rb') as file:
                    active = pickle.load(file)
                pg.setActiveVertices(active)
            else:
                pg.computeActiveVertices()
            refCache = None

        logging.info('Computing mesh parts')
        p = pg.defaultPose
        pg.setPose(p)
        neutral = pg.getVertices()
        active = pg.active
        faces = pg.getFaces()
        uvIdx = pg.getUVIndex()
        uv = pg.getUVs()
        uv = uv[uvIdx]
        pose,m = pg.generateBatch(n=5)
        parts = computeParts(uv,faces)

        # Remove any vertices that aren't included in parts
        if np.any(parts==-1):
            logging.info('Removing '+str(np.sum(parts==-1))+' vertices and recomputing mesh parts')
            activeIndex = np.arange(len(active))[active]
            active[activeIndex[parts==-1]] = False
            pg.setActiveVertices(active)

            p = pg.defaultPose
            pg.setPose(p)
            neutral = pg.getVertices()
            active = pg.active
            faces = pg.getFaces()
            uvIdx = pg.getUVIndex()
            uv = pg.getUVs()
            uv = uv[uvIdx]
            pose,m = pg.generateBatch(n=5)
            parts = computeParts(uv,faces)
            
    finally:
        pg.close()

    vCharts = parts
    for i in range(np.max(parts)+1):
        p = uv[vCharts==i]
        minVal,maxVal = np.min(p,0),np.max(p,0)
        uv[vCharts==i] = (p-minVal)/(maxVal-minVal)

    if refCache is None:
        vp = np.ones(np.max(parts)+1,dtype='bool')
    else:
        vp = refCache['visibleParts']

    data = dict(neutral=neutral,active=active,faces=faces,p=pose,m=m,uv=uv,vCharts=vCharts,originalFaces=faces,visibleParts=vp)
    with open(args.cacheFile,'wb') as file:
        pickle.dump(data,file)

if __name__=='__main__':
    main()

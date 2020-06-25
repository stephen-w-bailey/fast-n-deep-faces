import numpy as np

def buildFaceAdjList(faces):
    adj = [[] for _ in range(np.max(faces)+1)]
    for i,f in enumerate(faces):
        for v in f:
            adj[v].append(i)
    maxLen = np.max([len(a) for a in adj])
    for a in adj:
        while len(a) < maxLen:
            a.append(-1)
    adj = np.asarray(adj)
    return adj

def areaWeightedError(source,target,faces,adjList):
    v = source[faces.reshape(-1)].reshape((-1,3,3))
    a,b,c = v[:,0],v[:,1],v[:,2]
    ab,ac = b-a,c-a
    cross = np.cross(ab,ac)
    areaMesh = np.sqrt(np.sum(np.square(cross),-1))/2
    areaMesh = np.concatenate((areaMesh,[0]),axis=0)
    adjCount = np.sum(adjList>-1,-1)

    weights = areaMesh[adjList.reshape(-1)].reshape((len(source),adjList.shape[-1]))
    weights = np.sum(weights,-1)/adjCount

    diff = np.sqrt(np.sum(np.square(source-target),-1))
    error = np.sum(diff*weights)/np.sum(weights)

    return error

# Return a mask for degenerate triangles
def areaMask(v,faces):
    v = v[faces.reshape(-1)].reshape((-1,3,3))
    a,b,c = v[:,0],v[:,1],v[:,2]
    ab,ac = b-a,c-a
    cross = np.cross(ab,ac)
    mag = np.sqrt(np.sum(np.square(cross),-1))
    return mag>1e-6

def normal(v,faces):
    v = v[faces.reshape(-1)].reshape((-1,3,3))
    a,b,c = v[:,0],v[:,1],v[:,2]
    ab,ac = b-a,c-a
    cross = np.cross(ab,ac)
    mag = np.sqrt(np.sum(np.square(cross),-1))
    n = cross / mag[...,np.newaxis]
    return n

def normalError(source,target,faces,mask=None,returnMean=True):
    sourceNormal = normal(source,faces)
    targetNormal = normal(target,faces)
    if mask is not None:
        sourceNormal = sourceNormal[mask]
        targetNormal = targetNormal[mask]
    dot = np.sum(sourceNormal*targetNormal,-1)
    dot = dot[np.logical_not(np.isnan(dot))]
    dot = np.clip(dot,-1,1)
    angle = np.arccos(dot)
    angle = angle*180/np.pi
    if returnMean:
        return np.mean(angle)
    else:
        return angle
    

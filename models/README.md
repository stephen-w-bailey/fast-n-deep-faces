# Instructions:

### New character setup:

* Identify character controls and save as list in `controls.txt`
* Identify meshes in Maya scene to be approximated and save as list in `geo.txt`

## Running Maya server:

* Ensure numpy is installed in Maya's Python interpreter
* Run Maya and open the desired rig file.
* Setup working directories. In Maya's Script Editor, run:
```
	import sys
	sys.path.append("/full/path/to/top/level/of/this/repository")
	import os
	os.chdir("/full/path/to/top/level/of/this/repository")
```
* Import the PoseGenerator script in Maya and run it. In Maya's Script Editor, run:
```
	import PoseGenerator
	pg=PoseGenerator.PoseGeneratorServer('models/controls.txt','models/geo.txt')
	pg.generator.computeActiveVertices()
	pg.startServer()
```
* While the server is running, the Maya client is blocked and you won't be able to interact with it.
* The server exits whenever a client disconnects, restart the server by running `pg.startServer()` again

## Training the approximation model

The following instructions are not run from within Maya. The rest of the instructions are run from the command line.

### Create cache file:
* The cache file information about mesh topology, mesh segments, and rig controls
* Ensure that the PoseGenerator server is running (`pg.startServer()` in Maya)
* Run: 
```
python3 buildCache.py --cacheFile models/cache.pkl --controlFile models/controls.txt --geoFile models/geo.txt --activeVertices models/active_vertices.pkl
```
* --cacheFile is where you would like to save the cache
* The activeVertices parameter is optional, and it indicates which vertices on the mesh to include in the approximation

### Create rigid cache file:
* The rigid components need to be identified and attached to nonrigidly deformed vertices in the mesh
* Run: 
```
python3 computeRigidAttachment.py --configFile models/base_model_leaky.yaml --outputFile models/rigid_parts.pkl
```
* This script will modify the chace file spefified in the config file to indicate nonrigidly deformed vertices and will output a file containing information for attaching rigid components to vertices

### Determing rig control influence
* Ensure that the PoseGenerator server is running (`pg.startServer()` in Maya)
* Run:
```
python3 cnnModel/computeParameterMasks.py --configFile models/base_model_leaky.yaml --outputFile models/cache.pkl
```
* This will update your cache with rig control influence information

### Train the coarse model
* Ensure that the PoseGenerator server is running (`pg.startServer()` in Maya)
* Run: 
```
python3 cnnModel/trainModel.py --configFile models/base_model_leaky.yaml --checkpoint models/base_model_leaky
```

### Identify refinement segments
* Identify which mesh segment from the coarse model requires a refinement model (in the case of Ray, this is segment 0)
* Ensure that the PoseGenerator server is running (`pg.startServer()` in Maya)
* Run:
```
python3 uv/getMappingError.py --configFile models/base_model_leaky.yaml --outputError models/mappingError.pkl --partIndex 0

python3 uv/autoDetectRefinement.py --configFile models/base_model_leaky.yaml --errorFile models/mappingError.pkl --partIndex 0 --outputFile models/cache_refined.pkl --k 3 --errorCutoff 0.001
```

### Train refinement model
* Ensure that the PoseGenerator server is running (`pg.startServer()` in Maya)
* Run:
```
python3 cnnModel/poseAnalysis.py --configFile models/refine_model_leaky.yaml --outputFile models/cache_refined.pkl

python3 cnnModel/trainModelRefine.py --configFile models/refine_model_leaky.yaml --checkpoint models/refine_model_leaky
```

### Test model
* Ensure that the PoseGenerator server is running (`pg.startServer()` in Maya)
* Run:
```
python3 cnnModel/testModel.py --configFile models/refine_model_leaky.yaml --checkpoint models/refine_model_leaky --animFile models/test_poses.pkl
```
* L1, L2, and normal errors are printed to console
* If the file `models/test_poses.pkl` exists, the poses in that file will be evaluated
* If the file doesn't exist, the script willl generate random poses and save those to the file

### Time model
* PoseGenerator does not need to be running
* Run:
```
python3 cnnModel/timeModelFull.py --configFile models/refine_model_leaky.yaml --checkpoint models/refine_model_leaky --n 100 --sampleFile models/test_poses.pkl
```
* To run on CPU only, set the environment variable `CUDA_VISIBLE_DEVICES=""`

### Visualize Resulting Approximation
* PoseGenerator does not need to be running
* Run:
```
python3 visualizeApproximation.py --configFile models/refine_model_leaky.yaml --checkpoint models/refine_model_leaky
```
* Use the rig control sliders to see how they affect the resulting rig


## Training the IK Model

### Identify control points
* No script is available for picking control points
* Control point list is provided in `ik_control_points.pkl` for the Ray rig
* File specifies the face on the mesh and the barycentric coordinates for the point on the face

### Create Dataset
* This script generates poses at random and evaluates the mesh at those poses
* The file `ik_control_points_cache.pkl` stores the coordinates of the control points for each pose
* The file `ik_control_points_bins.pkl` stores the randomly generated poses as well as bin information for dataset balancing
* Ensure that the PoseGenerator server is running (`pg.startServer()` in Maya)
* Run:
```
python3 binData.py --configFile models/base_model_leaky.yaml --cacheFile models/ik_control_points_cache.pkl --pointFile models/ik_control_points.pkl --outputFile models/ik_control_points_bins.pkl
```

### Assign rig parameters to control points
* The script identifies which control points move when each rig parameter is activated
* Some controls do nothing when the rig is in the default pose
* The file `controls_influence_test.txt` specifies a new default pose in which all controls change the mesh (e.g. activating wrinkles) when adjusted one at a time.
* The output file `ik_control_points_masked.pkl` contains a mask for each control point indicating which rig parameters influence the point
* Ensure that the PoseGenerator server is running (`pg.startServer()` in Maya)
* Run:
```
python3 ikModel/computeParameterMasks.py --configFile models/ik_control_model_leaky.yaml --pointFile models/ik_control_points.pkl --outputFile models/ik_control_points_masked.pkl --controlFile models/controls_influence_test.txt
```

### Train the model
* PoseGenerator does not need to be running
* Run:
```
python3 ikModel/trainModel.py --config models/ik_control_model_leaky.yaml --checkpoint models/ik_control_model_leaky
```

### IK posing
* Run:
```
python3 ikModel/ikControl.py --configFile models/ik_control_model_leaky.yaml --checkpoint models/ik_control_model_leaky --camPos 5
```
* Click and drag points in the orthographic projection window
* Toggle the refinement model with the checkbox at the lower left corner of the other window

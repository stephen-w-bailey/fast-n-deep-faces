# Fast And Deep Facial Deformations

<img src="images/teaser.png" width="900px"/>

Stephen W. Bailey, Dalton Omens, Paul DiLorenzo, and James F. O'Brien

SIGGRAPH 2020

This is the public source code for the paper "Fast and Deep Facial Deformations" (2020).

This code is provided under a BSD license allowing its free use.  However we request that where feasible and reasonable you acknowledge use of this code.  We also request that any scholarly publications deriving from this code or from the algorithms it describes please cite the following publication:

> Stephen W. Bailey, Dalton Omens, Paul Dilorenzo, and James F. O'Brien. "Fast and Deep Facial Deformations”. ACM Transactions on Graphics, 39(4):94:1–15, August 2020. Presented at SIGGRAPH 2020, Washington D.C.

Website: [http://graphics.berkeley.edu/papers/Bailey-FDF-2020-07](http://graphics.berkeley.edu/papers/Bailey-FDF-2020-07)

### Installation

This code is tested with Maya 2018 and Python 3.6 on Windows 10 and Ubuntu 16.04

As of the publication date, Maya runs Python 2.7 but the main bulk of our scripts are running independently of Maya in Python 3. You will need a numpy installation for MayaPy (2.7) and all modules listed in requirements.txt can be installed for Python 3.

[This link](https://forums.autodesk.com/t5/maya-programming/guide-how-to-install-numpy-scipy-in-maya-windows-64-bit/td-p/5796722) can be helpful to install numpy for maya.

Run `pip install -r requirements.txt` in the root directory to install required Python modules.

### Quickstart Guide

Pretrained models for [Ray](https://www.cgtarian.com/maya-character-rigs/download-free-3d-character-ray.html) are provided with this code.  You can evaluate the models without installing and running the Ray character rig in Maya.

* Download the compressed archive of the models [here](http://graphics.berkeley.edu/papers/Bailey-FDF-2020-07/models.zip)
* Unpack the contents of the archive in the `models` subdirectory
* Control the model through rig parameters by running the following:
```
python3 visualizeApproximation.py --configFile models/refine_model_leaky.yaml --checkpoint models/refine_model_leaky
```
* Control the model through IK control points by running the following:
```
python3 ikModel/ikControl.py --configFile models/ik_control_model_leaky.yaml --checkpoint models/ik_control_model_leaky --camPos 5
```

### Running the Code

Detailed instructions for training course and refinement models are in `models/README.md`.

### Citation

This citation is also included as a .bib file in the repository.

```
@article{Bailey:2020:FDF,
  note = {Presented at SIGGRAPH 2020, Washington D.C.},
  doi = {10.1145/3386569.3392397},
  title = {Fast and Deep Facial Deformations},
  journal = {ACM Transactions on Graphics},
  author = {Stephen W. Bailey and Dalton Omens and Paul Dilorenzo and James F. O'Brien},
  number = 4,
  month = aug,
  volume = 39,
  year = 2020,
  pages = {94:1--15},
  url = {http://graphics.berkeley.edu/papers/Bailey-FDF-2020-07/},
}
```
### Acknowledgments

We would like to thank DreamWorks Animation for granting us access to their resources and data during the development of this project.  We would also like to thank CGTarian Online School for developing and releasing the Ray character rig.

### License

```

Copyright (c) 2020, Stephen W. Bailey and Dalton Omens

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of {{ project }} nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

```

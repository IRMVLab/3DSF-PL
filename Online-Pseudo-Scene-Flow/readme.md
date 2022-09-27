 ## Introduction
The main purpose of this paper is to recover 3D scene flow from stereo images. The stereo images are represented by $I^l$ and $I^r$ respectively. As Fig. \ref{fig:Framework}, given a pair of stereo images, which contains reference frames $\{I_{t}^l, I_{t}^r\}$ and target frames $\{I_{t+1}^l, I_{t+1}^r\}$. Each image is represented by a matrix of dimension $H\times W\times 3$. Depth map $D_t$ at time $t$ is predicted by feeding the stereo image $\{I_{t}^l, I_{t}^r\}$ into the depth estimation network $D_{net}$. Each pixel value of $D$ represents the distance $d$ between a certain point in the scene and the left camera. Pseudo-LiDAR point cloud comes from back-projecting the generated depth map to a 3D point cloud. The 3D point coordinate $(x_w, y_w, z_w)$ in the pseudo-LiDAR point cloud $PL$ is calculated by pixel coordinates $(u, v)$, $d$ and camera intrinsics. $PL_t = \{c_{1,i}\in{\mathbb{R}^3}\}_{i=1}^{N_1}$ with $N_1$ points and PL_{t+1} = \{c_{2,j}\in{\mathbb{R}^3}\}_{j=1}^{N_2}$ with $N_2$ points are generated from the depth maps $\mathcal{D}_t$ and $\mathcal{D}_{t+1}$, where $c_{1,j}$ and $c_{2,j}$ are the 3D coordinates of the points. $\mathcal{PL}_t$ and $\mathcal{PL}_{t+1}$ are randomly sampled to $N$ points, respectively. The sampled pseudo-LiDAR point clouds are passed into the scene flow estimator $\mathcal{F}_{sf}$ to extract the scene flow vector $\mathcal{SF}_t=\{sf_i|i=1,2,\cdots,N\}$ for each 3D point in frame $t$.


## Installation
### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04)
* Python 3.6+
* PyTorch 1.0

### Install


a. Install the dependent python libraries like `easydict`,`tqdm`, `tensorboardX ` etc.

b. Build and install the `pointnet2_lib`, `iou3d`, `roipool3d` libraries by executing the following command:
```shell
sh build_and_install.sh
```


## Training
Currently, the two stages of PointRCNN are trained separately. Firstly, to use the ground truth sampling data augmentation for training, we should generate the ground truth database as follows:
```
python train.py config_train.yaml
```

## Inference
* To evaluate a single checkpoint, run the following command with `--ckpt` to specify the checkpoint to be evaluated:
```
python evaluate.py config_evaluate.yaml
```

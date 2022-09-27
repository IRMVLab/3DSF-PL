 ## Introduction
The main purpose of this paper is to recover 3D scene flow from stereo images. The stereo images are represented by $I^l$ and $I^r$ respectively. Given a pair of stereo images, which contains reference frames $\{I_{t}^l, I_{t}^r\}$ and target frames $\{I_{t+1}^l, I_{t+1}^r\}$. Each image is represented by a matrix of dimension $H\times W\times 3$. Depth map $D_t$ at time $t$ is predicted by feeding the stereo image $\{I_{t}^l, I_{t}^r\}$ into the depth estimation network $D_{net}$. Each pixel value of $D$ represents the distance $d$ between a certain point in the scene and the left camera. Pseudo-LiDAR point cloud comes from back-projecting the generated depth map to a 3D point cloud. The generated pseudo-LiDAR point cloud is passed into the scene flow estimator to learn a 3D scene flow for each visible point.


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
Download the <a href="https://pan.baidu.com/s/1USMcnw-hkkuP60-8P9pU8w?pwd=ck98">weights</a> of the pre-trained depth estimation network and change its path in the configuration file.
```
python train.py config_train.yaml
```

## Inference
* To evaluate a single checkpoint, run the following command with `--ckpt` to specify the checkpoint to be evaluated:
```
python evaluate.py config_evaluate.yaml
```

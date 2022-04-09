# LD-Net

LD-Net: A Lightweight Network for Real-Time
Self-Supervised Monocular Depth Estimation

# Abstract

Self-supervised monocular depth estimation from video sequences is promising for 3D environments perception.
However, most existing methods use complicated depth networks
to realize monocular depth estimation, which are often difficultly
applied to resource-constrained devices. To solve this problem, in
this letter, we propose a novel encoder-decoder-based lightweight
depth network (LD-Net). To solve this problem, in
this letter, we propose a novel encoder-decoder-based lightweight
depth network (LD-Net). Briefly speaking, the encoder is composed
of six efficient downsampling units and the Atrous Spatial Pyramid Pooling (ASPP) module. The decoder consists of some novel
upsampling units that adopt the sub-pixel convolutional layer (SP).
Experiments tested on the KITTI dataset show that the proposed
LD-Net can reach nearly 150 frames per second (FPS) on GPU,
and remarkably decreases the model parameters while maintaining
competitive accuracy compared with other state-of-the-art selfsupervised monocular depth estimation methods.



# Citing

```
@ARTICLE{9738438,
  author={Xiong, Mingkang and Zhang, Zhenghong and Zhang, Tao and Xiong, Huilin},
  journal={IEEE Signal Processing Letters}, 
  title={LD-Net: A Lightweight Network for Real-Time Self-Supervised Monocular Depth Estimation}, 
  year={2022},
  volume={29},
  pages={882-886},
  doi={10.1109/LSP.2022.3160656}}
```
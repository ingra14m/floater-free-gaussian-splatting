# Robust-Gaussian-Splatting
This repo uses the absolute value of the gradient of pixel pairs for GS to accumulate the gradient of each GS. Since the preprint paper [absGS](https://ty424.github.io/AbsGS.github.io/) did the same thing, therefore, you can also consider this repo as an unofficial implementation of [absGS](https://ty424.github.io/AbsGS.github.io/). 



Compared to [Pixel-GS](https://pixelgs.github.io/), our project can achieve the removal of floaters without significantly increasing the number of GS. In some scenarios where the point cloud distribution is good, it can reduce the number of point clouds. Compared to [Radsplat](https://arxiv.org/abs/2403.13806), our method does not require training zipnerf, and the training time on a 3090 is approximately 30 minutes.



## Dataset

In this project, you can use:

- synthetic dataset from [NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1), and [NSVF](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip)
- real-world dataset from [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) and [tandt_db](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

And the data structure should be organized as follows:

```shell
data/
├── NeRF
│   ├── Chair/
│   ├── Drums/
│   ├── ...
├── NSVF
│   ├── Bike/
│   ├── Lifestyle/
│   ├── ...
├── Mip-360
│   ├── bicycle/
│   ├── bonsai/
│   ├── ...
├── tandt_db
│   ├── db/
│   │   ├── drjohnson/
│   │   ├── playroom/
│   ├── tandt/
│   │   ├── train/
│   │   ├── truck/
```





## Run

### Environment

```shell
git clone https://github.com/ingra14m/robust-gaussian-splatting --recursive
cd robust-gaussian-splatting

conda create -n abs-gaussian-env python=3.8
conda activate abs-gaussian-env

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install dependencies
pip install -r requirements.txt
```



### Train

```shell
python train.py -s your/path/to/the/dataset -m your/path/to/save --eval
```



## Results

| Scene    | PSNR  | SSIM   | LPIPS  | Mem  | FPS  |
| -------- | ----- | ------ | ------ | ---- | ---- |
| bicycle  | 25.82 | 0.7989 | 0.1656 | 1441 | 66   |
| bonsai   | 32.41 | 0.9502 | 0.1608 | 258  | 170  |
| counter  | 29.22 | 0.9187 | 0.1687 | 261  | 125  |
| garden   | 27.95 | 0.8799 | 0.0934 | 971  | 65   |
| kitchen  | 31.91 | 0.9351 | 0.1081 | 434  | 99   |
| room     | 31.78 | 0.9331 | 0.1750 | 416  | 114  |
| stump    | 27.3  | 0.7976 | 0.1848 | 1043 | 103  |
| flower   | 21.84 | 0.6495 | 0.2629 | 888  | 105  |
| treehill | 22.39 | 0.6475 | 0.2697 | 1087 | 87   |
| Average  | 27.85 | 0.8345 | 0.1765 | 755  | 104  |



## Methods

```c++
// vanilla gradients for densification
atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

// abs gradients for densification
atomicAdd(&dL_dmean2D_densify[global_id].x, fabsf(dL_dG * dG_ddelx * ddelx_dx));
atomicAdd(&dL_dmean2D_densify[global_id].y, fabsf(dL_dG * dG_ddely * ddely_dy));
```



## BibTex

This idea is the same as [absGS](https://arxiv.org/abs/2404.10484) and [Gaussian Opacity Fields](https://arxiv.org/abs/2404.10772). The difference is that we have set the `densify_grad_threshold` to 0.0005, and all other parameters are used as in vanilla 3D-GS. If you find this project useful, please don't forget to cite these two awesome papers. 

```shell
@article{ye2024absgs,
  title={AbsGS: Recovering Fine Details for 3D Gaussian Splatting},
  author={Ye, Zongxin and Li, Wenyu and Liu, Sidun and Qiao, Peng and Dou, Yong},
  journal={arXiv preprint arXiv:2404.10484},
  year={2024}
}

@article{Yu2024GOF,
  author    = {Yu, Zehao and Sattler, Torsten and Geiger, Andreas},
  title     = {Gaussian Opacity Fields: Efficient High-quality Compact Surface Reconstruction in Unbounded Scenes},
  journal   = {arXiv:2404.10772},
  year      = {2024},
}
```

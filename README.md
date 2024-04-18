# Robust-Gaussian-Splatting

![Teaser image](assets/teaser.png)

This repository contains the official authors implementation associated with the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering", which can be found [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). We further provide the reference images used to create the error metrics reported in the paper, as well as recently created, pre-trained models. 



## Results





## BibTex

This idea is the same as [absGS](https://arxiv.org/abs/2404.10484) and [Gaussian Opacity Fields](https://arxiv.org/abs/2404.10772). The only difference is that we have set the `densify_grad_threshold` to 0.0005, and all other parameters are used as in vanilla 3D-GS. If you find this project useful, please don't forget to cite these two awesome papers. 

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

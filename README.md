# TransSfP
Repository for paper "Transparent Shape from a Single View Polarization Image"

We provide dataset and codes of TransSfP.

[Paper](https://arxiv.org/pdf/2204.06331.pdf)|[Code](https://github.com/shaomq2187/TransSfP)[|Dataset](https://pan.baidu.com/s/1LkuLsu_ThLxUsuI8NM22gQ)

## Abstract

​	This paper presents a learning-based method for transparent surface estimation from a single view polarization image. Existing shape from polarization(SfP) methods have the difficulty in estimating transparent shape since the inherent transmission interference heavily reduces the reliability of physics-based prior. To address this challenge, we propose the concept of physics-based prior, which is inspired by the characteristic that the transmission component in the polarization image has more noise than reflection. The confidence is used to determine the contribution of the interfered physics-based prior. Then, we build a network(TransSfP) with multi-branch architecture to avoid the destruction of relationships between different hierarchical inputs. To train and test our method, we construct a dataset for transparent shape from polarization with paired polarization images and ground-truth normal maps. Extensive experiments and comparisons demonstrate the superior accuracy of our method. Our codes and data are provided in the supplements.

![image-20221129223722286](figures\image-20221129223722286.png)

## Train

Run the following command to train the TransSfP model:

`python train.py -batch_size=5 -dataset_dir='/media/disk/dataset/TransSfP' -code_dir='/media/disk/code'`

Pretrained model will be provided later.

## Dataset

<img src="https://raw.githubusercontent.com/s1752729916/githubsshaomq.github.iogithub/master/image-20220527171332530.png" alt="image-20220527171332530" style="zoom:50%;" />

Our dataset consists of two parts, a synthetic dataset and a real dataset.  The synthetic dataset contains 13 objects, each object has 72 sets of polarized images (different views) and a total of 936 sets. 
The real-world data set contains 10 objects, each object contains 10-60 sets, a total of 486 sets.

Download：[Baidu Netdisk (sbqn) ](https://pan.baidu.com/s/1LkuLsu_ThLxUsuI8NM22gQ ) |Google Drive

The format and organization of the synthetic dataset is similar to the real-world dataset. Their formats and organizations are listed as follows: 

```
1、Overview of our dataset
├── TransSfP
    ├── real-world
    │   ├── bird-back
    │   ├── cat-back
    │   ├── cat-front
    │   ├── hemi-sphere-big
    │   ├── hemi-sphere-small
    │   ├── middle-round-cup
    │   ├── middle-square-cup
    │   ├── middle-white-cup
    │   ├── tiny-cup
    │   └── tiny-cup-edges
    └── synthetic
        ├── armadillo-back
        ├── armadillo-front
        ├── bear-front
        ├── bun-zipper-back
        ├── bun-zipper-front
        ├── cow-back
        ├── cow-front
        ├── dragon-vrip
        ├── happy-vrip-back
        ├── happy-vrip-front
        ├── middle-round-cup
        ├── pot-back
        └── pot-front
 
2、Real-world dataset structure
├── real-world
    ├── bird-back
        ├── I-0              					# polarization image with angle of polarizer 0°, (1232x1028,UInt8)       
        ├── I-45								# polarization image with angle of polarizer 45°, (1232x1028,UInt8)   
        ├── I-90								# polarization image with angle of polarizer 90°, (1232x1028,UInt8)  		
        ├── I-135								# polarization image with angle of polarizer 135°, (1232x1028,UInt8)  			
        ├── I-sum								# intensity image, (1232x1028,UInt8)
        ├── masks								# foreground mask  (1232x1028,UInt8)
        ├── normals-png							# ground truth surface normals,[-1,1] to [0,255] (1232x1028x3,UInt8)
        ├── params								
        │   ├── AoLP							# angle of linear polarization, [0°,180°] to [0,255] (1232x1028,UInt8)
        │   └── DoLP							# degree of linear polarization, [0,1] to [0,255] (1232x1028,UInt8)
        └── synthesis-normals					# physics-based priors calculated from specular reflection model, [-1,1] to [0,255] (1232x1028x3,UInt8)
            ├── synthesis-normal-0				
            ├── synthesis-normal-1
            ├── synthesis-normal-2
            └── synthesis-normal-3

3、Synthetic dataset structure
└── synthetic
    ├── armadillo-back
        ├── I-0									# polarization image with angle of polarizer 0°, (1232x1028,UInt8)   
        ├── I-45								# polarization image with angle of polarizer 45°, (1232x1028,UInt8)
        ├── I-90								# polarization image with angle of polarizer 90°, (1232x1028,UInt8)
        ├── I-135								# polarization image with angle of polarizer 135°, (1232x1028,UInt8)  
        ├── I-sum								# intensity image, (1232x1028,UInt8)
        ├── masks								# foreground mask, (1232x1028,UInt8)
        ├── normals-exr							# ground truth surface normals, (1232x1028x3,double)
        ├── normals-png							# ground truth surface normals,[-1,1] to [0,255] (1232x1028x3,UInt8)
        ├── params
        │   ├── AoLP							# angle of linear polarization, [0°,180°] to [0,255] (1232x1028,UInt8)
        │   └── DoLP							# degree of linear polarization, [0,1] to [0,255] (1232x1028,UInt8)        
        └── synthesis-normals					# physics-based priors calculated from specular reflection model, [-1,1] to [0,255] (1232x1028x3,UInt8)
            ├── synthesis-normal-0				
            ├── synthesis-normal-1
            ├── synthesis-normal-2
            └── synthesis-normal-3
```

## Citation

If you find our work useful in your research, please consider citing:

```tex
@article{mingqi2022transparent,
  title={Transparent Shape from Single Polarization Images},
  author={Mingqi, Shao and Chongkun, Xia and Zhendong, Yang and Junnan, Huang and Xueqian, Wang},
  journal={arXiv preprint arXiv:2204.06331},
  year={2022}
}
```


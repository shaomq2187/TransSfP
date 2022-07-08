# TransSfP
Repository for paper "Transparent Shape from Single Polarization Images"

## Abstract

This paper presents a data-driven approach for transparent shape from polarization. Due to the inherent high transmittance, the previous shape from polarization(SfP) methods based on specular reflection model have difficulty in estimating transparent shape, and the lack of datasets for transparent SfP also limits the application of the data-driven approach. Hence, we construct the transparent SfP dataset which consists of both synthetic and real-world datasets. To determine the reliability of the physics-based reflection model, we define the physics-based prior confidence by exploiting the inherent fault of polarization information, then we propose a multi-branch fusion network to embed the confidence. Experimental results show that our approach outperforms other SfP methods. Compared with the previous method, the mean and median angular error of our approach are reduced from $19.00^\circ$ and $14.91^\circ$ to $16.72^\circ$ and $13.36^\circ$, and the accuracy $11.25^\circ, 22.5^\circ, 30^\circ$ are improved from $38.36\%, 77.36\%, 87.48\%$ to $45.51\%, 78.86\%, 89.98\%$, respectively.

## Dataset

<img src="https://raw.githubusercontent.com/s1752729916/githubsshaomq.github.iogithub/master/image-20220527171332530.png" alt="image-20220527171332530" style="zoom:50%;" />

TransSfP dataset consists of two parts, a synthetic dataset and a real dataset.  The synthetic dataset contains 13 objects, each object has 72 sets of polarized images (different views) and a total of 936 sets. 
The real-world data set contains 10 objects, each object contains 10-60 sets, a total of 486 sets.

Download：[Baidu Netdisk (sbqn) ](https://pan.baidu.com/s/1LkuLsu_ThLxUsuI8NM22gQ ) |Google Drive

The format and organization of the synthetic dataset is similar to the real-world dataset. Their formats and organizations are listed as follows: 

```
1、Overview of TransSfP
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


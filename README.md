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



## Analysis of Physics-based Prior Confidence

![image-20220528001653910](https://raw.githubusercontent.com/s1752729916/githubsshaomq.github.iogithub/master/image-20220528001653910.png)

The key idea of our paper is the physics-based prior confidence, which is inspired by the observation that the areas with high transmittance have more noise in the AoLP map. Here we will analyze the reason for this phenomenon in detail.

First give our conclusion: **the noise in AoLP map is caused by the uncertainty of the dominance of the specular and diffuse components of the background.**

The specular reflection and diffuse reflection models are often used in SfP problems to calculate the surface normals. Here we only care about the AoLP. The specular(![psi_s](https://latex.codecogs.com/svg.image?\psi_s) and diffuse(![psi_d](https://latex.codecogs.com/svg.image?\psi_d)) components are related to the azimuth angle ![phi](https://latex.codecogs.com/svg.image?\phi) of the surface normal as follows:

![CodeCogsEqn (3)](https://raw.githubusercontent.com/s1752729916/githubsshaomq.github.iogithub/master/CodeCogsEqn%20(3).svg)

The reflected light ![I_t](https://latex.codecogs.com/svg.image?I_t(\theta_{pol}))  of the background consists of specular ![I_s](https://latex.codecogs.com/svg.image?I_s(\theta_{pol})) and diffuse ![I_d](https://latex.codecogs.com/svg.image?I_d(\theta_{pol})) components:

![CodeCogsEqn (2)](https://raw.githubusercontent.com/s1752729916/githubsshaomq.github.iogithub/master/CodeCogsEqn%20(2).svg)

where  ![theta_pol](https://latex.codecogs.com/svg.image?\theta_{pol})  represents the angle of polarizer,  ![1](https://latex.codecogs.com/svg.image?I_s,I_d) are the average intensity of the specular and diffuse components.  ![1](https://latex.codecogs.com/svg.image?\rho_s,\rho_d) are the DoLP of the specular and diffuse components, respectively. 

Therefore, the AoLP of the background reflected light ![1](https://latex.codecogs.com/svg.image?\psi_t) can be written as:

![CodeCogsEqn (4)](https://raw.githubusercontent.com/s1752729916/githubsshaomq.github.iogithub/master/CodeCogsEqn%20(4).svg)

When the dominance of diffuse and specular components is uncertain, the signs of ![1](https://latex.codecogs.com/svg.image?I_s\rho_s-I_d\rho_d) of adjacent points will change frequently, so the observed polarization angles of adjacent pixels will generate ![1](https://latex.codecogs.com/svg.image?\frac{&space;\pi}{2}) phase shift(the well-known ![1](https://latex.codecogs.com/svg.image?\frac{&space;\pi}{2})- **ambiguity**), this is why the background has more noise in the AoLP map.

 Since the surface of the transparent object is smooth enough, it can be assumed that the observed light of the transparent object only contains specular reflection ![1](https://latex.codecogs.com/svg.image?I_r(\theta_{pol})) and transmission ![1](https://latex.codecogs.com/svg.image?I_t(\theta_{pol})):

![CodeCogsEqn](https://raw.githubusercontent.com/s1752729916/githubsshaomq.github.iogithub/master/CodeCogsEqn.svg)



 ![1](https://latex.codecogs.com/svg.image?I_{r0}) is the light intensity value corresponding to the incident light. The diffuser in the experimental setup of this paper ensures that ![1](https://latex.codecogs.com/svg.image?I_{r0}) in all directions have the same value,  ![1](https://latex.codecogs.com/svg.image?T) is the transmission coefficient,  ![1](https://latex.codecogs.com/svg.image?\rho_r,&space;\psi_r) are the DoLP and AoLP of the specular reflection component on the surface of the transparent object, respectively,  ![1](https://latex.codecogs.com/svg.image?\rho) and ![1](https://latex.codecogs.com/svg.image?\psi) are DoLP and AoLP of the transparent object surface actually observed by the camera.

Rewrite the above formula into the following form:

![CodeCogsEqn (1)](https://raw.githubusercontent.com/s1752729916/githubsshaomq.github.iogithub/master/CodeCogsEqn%20(1).svg)



We can get ![1](https://latex.codecogs.com/svg.image?\psi) as follows:

![CodeCogsEqn (5)](https://raw.githubusercontent.com/s1752729916/githubsshaomq.github.iogithub/master/CodeCogsEqn%20(5).svg)



Since the value of ![1](https://latex.codecogs.com/svg.image?I_{r0}) is determined by the light source,  ![1](https://latex.codecogs.com/svg.image?I_{t0}) is determined by the reflection of the background, usually ![1](https://latex.codecogs.com/svg.image?\frac{I_{r0}}{I_{t0}}\approx10) , and the value of $\frac{\rho_r}{\rho_t}$ is related to the transmission coefficient  ![1](https://latex.codecogs.com/svg.image?T), but generally greater than 1. When ![1](https://latex.codecogs.com/svg.image?T) is small, the transmission term can be ignored , that is, ![1](https://latex.codecogs.com/svg.image?\psi=\psi_r); when ![1](https://latex.codecogs.com/svg.image?T\rightarrow1) , ![1](https://latex.codecogs.com/svg.image?\psi)  will be disturbed by ![1](https://latex.codecogs.com/svg.image?\psi_t) and appear noise as shown in above Figure.

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


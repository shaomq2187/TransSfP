TransSfP:
TranSfP is a dataset for transparent shape from polarization(SfP). We hope this dataset will drive more exciting works on transparent SfP.

Data Summary:
TransSfP dataset consists of two parts, a synthetic dataset and a real dataset. 
The synthetic dataset contains 13 objects, each object has 72 sets of polarized images (different views) and a total of 936 sets. 
The real-world data set contains 10 objects, each object contains 10-60 sets, a total of 486 sets.

Dataset Organizations:
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
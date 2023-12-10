# OCTDL: Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods

Framework for 2D classification of OCT images.

Optical coherence tomography (OCT) is an emerging technology for performing high-resolution cross-sectional imaging. 
It utilizes the interferometry concept to create a cross-sectional map of the retina. 
OCT images are two-dimensional data sets that represent the optical backscattering in a cross-sectional plane 
through the tissue. 
OCT allows non-secondary visualization of various structures of the eye, including the retina, vitreous body,
and choroid, and to detect pathological changes in them. 
The study of these images is essential for the diagnosis, treatment, and monitoring of various eye diseases.

The dataset consists of the following categories and images:

- Age-Related Macular Degeneration - 885 images;
- Diabetic Macular Edema - 143 images;
- Epiretinal Membrane- 133 images;
- Normal - 284 images;
- Retinal Artery Occlusion - 22 images;
- Retinal Artery Occlusion - 93 images;
- Vitreomacular Interface Disease - 58 images.


> Mikhail Kulyabin, Aleksei Zhdanov, Vasilii Borisov, Mikhail Ronkin, Stepichev Andrey, Kuznetsova Anna, Nikiforova Anastasia, Bogachev Alexander, Korotkich Sergey, June 9, 2023, "OCTDL: Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods", 
> IEEE Dataport, doi: [[link](https://dx.doi.org/10.21227/fpvs-8n55)]

Requirements:
- pytorch
- torchvision
- torcheval
- timm
- tqdm
- munch
- packaging
- tensorboard
- omegaconf
- opencv-python
- hydra-core
- scikit-learn



## How to use
In this work we use folder-form dataset structure:
```
├── dataset
    ├── train
        ├── class1
            ├── image1.jpg
            ├── ...
        ├── class2
        ├── class3
        ├── ...
    ├── val
    ├── test
```
Unzip the archives to `./OCTDL_folder` and use `preprocessing.py` for image preprocessing and dataset preparation:
```
python preprocessing.py
```
Optional arguments:
```
--dataset_folder', type=str, default='./OCTDL_folder', help='path to dataset folder')
--output_folder', type=str, default='./dataset', help='path to output folder')
--crop_ratio', type=int, default=1, help='central crop ratio of image')
--image_dim', type=int, default=512, help='final dimensions of image')
--val_ratio', type=float, default=0.1, help='validation size')
--test_ratio', type=float, default=0.2, help='test size')
--padding', type=bool, default=False, help='padding to square')
```

Training:

```shell
python main.py
```

Optional arguments:
```
-c yaml_file      Specify the config file (default: configs/default.yaml)
-p                Print configs before training
```

Use Hydra to run multiple jobs with different arguments with a single command
```
python3 main.py train.network=resnet50,vgg16 -m
```

## Resources

> Huang, Y., Lin, L., Cheng, P., Lyu, J., Tam, R. and Tang, X., 2023. Identifying the key components in ResNet-50 for diabetic retinopathy grading from fundus images: a systematic investigation. Diagnostics, 13(10), p.1664. [[link](https://www.mdpi.com/2075-4418/13/10/1664)]


## Citation
```
@data{fpvs-8n55-23,
doi = {10.21227/fpvs-8n55},
url = {https://dx.doi.org/10.21227/fpvs-8n55},
author = {Kulyabin, Mikhail and Zhdanov, Aleksei and Borisov, Vasilii and Ronkin, Mikhail and Andrey, Stepichev and Anna, Kuznetsova and Anastasia, Nikiforova and Alexander, Bogachev and Sergey, Korotkich},
publisher = {IEEE Dataport},
title = {OCTDL: Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods},
year = {2023} } 
```

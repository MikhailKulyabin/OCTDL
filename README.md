# Optical coherence tomography dataset classification

Framework for 2D classification of the OCT dataset.

Optical coherence tomography (OCT) is an emerging technology for performing high-resolution cross-sectional imaging. 
It utilizes the interferometry concept to create a cross-sectional map of the retina. 
OCT images are two-dimensional data sets that represent the optical backscattering in a cross-sectional plane 
through the tissue. 
OCT allows non-secondary visualization of various structures of the eye, including the retina, vitreous body,
and choroid, and to detect pathological changes in them. 
The study of these images is essential for the diagnosis, treatment, and monitoring of various eye diseases.

The database consists of the following categories and images:

- Age-related Macular Degeneration (AMD): 1231
- Diabetic Macular Edema (DME): 147
- Epiretinal Membrane (ERM): 155
- Normal (NO): 332
- Retinal Artery Occlusion (RAO): 22
- Retinal Vein Occlusion (RVO): 101
- Vitreomacular Interface Disease (VID): 76

> Mikhail Kulyabin, Aleksei Zhdanov, Vasilii Borisov, Mikhail Ronkin, Stepichev Andrey, Kuznetsova Anna, Nikiforova Anastasia, Bogachev Alexander, Korotkich Sergey, June 9, 2023, "Optical coherence tomography dataset", 
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
Unzip the archive to the directory and use `preprocessing.py` for image preprocessing and dataset preparation:
```
python preprocessing.py
```
Optional arguments:
```
--dataset_folder', type=str, default='./OCT_dataset', help='path to dataset folder')
--labels_path', type=str, default='./OCTDL_dataset/labels.csv', help='path to labels.csv'
--output_folder', type=str, default='./dataset', help='path to output folder')
--crop_ratio', type=int, default=1, help='central crop ratio of image')
--image_dim', type=int, default=512, help='final dimensions of image')
--val_ratio', type=float, default=0.15, help='validation size')
--test_ratio', type=float, default=0.25, help='test size')
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



## Resources

> Huang, Y., Lin, L., Cheng, P., Lyu, J., Tam, R. and Tang, X., 2023. Identifying the key components in ResNet-50 for diabetic retinopathy grading from fundus images: a systematic investigation. Diagnostics, 13(10), p.1664. [[link](https://www.mdpi.com/2075-4418/13/10/1664)]


## Citation
```
@data{fpvs-8n55-23,
doi = {10.21227/fpvs-8n55},
url = {https://dx.doi.org/10.21227/fpvs-8n55},
author = {Kulyabin, Mikhail and Zhdanov, Aleksei and Borisov, Vasilii and Ronkin, Mikhail and Andrey, Stepichev and Anna, Kuznetsova and Anastasia, Nikiforova and Alexander, Bogachev and Sergey, Korotkich},
publisher = {IEEE Dataport},
title = {Optical coherence tomography database},
year = {2023} }
```

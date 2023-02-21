# Parallel-Gan_pytorch
This is the official repo for our work 'SAR-to-Optical Image Translation with Hierarchical Latent Features' (TGRS 2022).  
[Paper](https://https://ieeexplore.ieee.org/document/9864654)  
Citation information:  
```
@ARTICLE{9864654,
  author={Wang, Haixia and Zhang, Zhigang and Hu, Zhanyi and Dong, Qiulei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SAR-to-Optical Image Translation With Hierarchical Latent Features}, 
  year={2022},
  volume={60},
  number={},
  pages={1-12},
  doi={10.1109/TGRS.2022.3200996}}
```

## Setup
We built and ran the repo with CUDA 10.2, Python 3.7.6, and Pytorch 1.4.0. For using this repo, we recommend creating a virtual environment by [Anaconda](https://www.anaconda.com/products/individual). Please open a terminal in the root of the repo folder for running the following commands and scripts.
```
conda env create -f environment.yml
conda activate pytorch170cu10
```

## Pre-trained models
|Model Name|Dataset(s)|PSNR Rel.|SSIM Rel.|SAM|FID|
|----------|----------|--------|-------|----|-------|
|Reconstruction network G_R [Baidu](https://pan.baidu.com/s/10udDJaswX44SXg11a-Vn8Q?pwd=1234)/[Google](https://drive.google.com/file/d/1wpOj39KGgKGHpG_Z_MRGAUgyxF__3sVz/view?usp=sharing)|K|29.9690|0.7043|1.2850|70.8376|
|Translation network G_T [Baidu](https://pan.baidu.com/s/1Flie7J8K04oWeu0k8zCpqw?pwd=1234)/[Google](https://drive.google.com/file/d/1RxCJ6lz6MpeHIPLNFmm1hJikeDUOBXu8/view?usp=sharing)|K|22.4651|0.3871|2.7256|102.1934|

* **code for all the download links of Baidu is `1234`**
## Quickly Start
To predict optical image from the corresponding images, please firstly download the pretrained model from the column named `Model Name` in the above table. Some test image from SpaceNet Dataset(MSAW dataset) has been put in the dir "./dataset/SpaceNet/". After download the translation or reconstruction model into the dir:"./checkpoints/name/latest_net_G_trans.pth" or "./checkpoints/name/latest_net_G_recon.pth", you can test the models by:
```
CUDA_VISIBLE_DEVICES=0 python\
 test.py\
 --name trans <name of the experiment. It decides where to store samples and models. And the path of download models should be like:"./checkpoints/name/model.pth">\
 --net translation <which net you want to test(reconstruction or translation)>\
 ```
## Data preparation
MSAW Dataset can be download by
```
aws s3 ls s3://spacenet-dataset/spacenet/SN6_buildings/
```
And the QXS-SAROPT and SEN1-2 Dataset can be downloaded by the link1(https://github.com/yaoxu008/QXS-SAROPT) and link2(https://mediatum.ub.tum.de/1436631)
#### Set Data Path
Please change the "--dataroot" of the base_options.py to your dataset path. 
the folder for each dataset should be organized like(A for optical image, B for SAR image):
```
<root of Datatset>
|---train_A
|   |---1_A.tif
|   |---2_A.tif
|   |---...
|---train_B
|   |---1_B.tif
|   |---2_B.tif
|   |---...
|---test_A
|   |---1_A.tif
|   |---2_A.tif
|   |---...
|---test_B
|   |---1_B.tif
|   |---2_B.tif
|   |---...
```

for different dataset, you should set different "--input_nc". For MSAW Dataset with 4 channels SAR image:"--input_nc 4", for other datasets with 3 channels:"--input_nc 3".
## Training and Testing
You could use the following commands for training the Parallal-GAN on SpaceNet. As mentioned in our paper, we trained the reconstruction network and translation network respectively. The training is devideded into two stages:
```
# train reconstruction network at stage1
CUDA_VISIBLE_DEVICES=0 python\
 train.py\
 --name recon\
 --net reconstruction
# and test the reconstruction network:
 CUDA_VISIBLE_DEVICES=0 python\
 test.py\
 --name recon\
 --net reconstruction
```
After finishing the stage1, please copy the path to the last saved model `./checkpoints/name/latest_net_G_recon.pth` into `load_path` of parallel_gan_model.py, and use the following command to continue the training:
```
# train translation network at stage2
CUDA_VISIBLE_DEVICES=0 python\
 train.py\
 --name trans\
 --net translation
# and test the translation network:
 CUDA_VISIBLE_DEVICES=0 python\
 train.py\
 --name trans\
 --net translation
 ```

## Acknowledgment
Some of this repo come from [Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

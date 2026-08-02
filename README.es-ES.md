# Parallel-Gan_pytorch
Este es el repositorio oficial de nuestro trabajo 'SAR-to-Optical Image Translation with Hierarchical Latent Features' (TGRS 2022).  
[Artículo](https://https://ieeexplore.ieee.org/document/9864654)  
Información de cita:  
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

## Configuración
Construimos y ejecutamos el repositorio con CUDA 10.2, Python 3.7.6 y Pytorch 1.4.0. Para usar este repositorio, recomendamos crear un entorno virtual con [Anaconda](https://www.anaconda.com/products/individual). Abra una terminal en la raíz de la carpeta del repositorio para ejecutar los siguientes comandos y scripts.
```
conda env create -f environment.yml
conda activate ph
```

## Modelos preentrenados
|Nombre del modelo|Conjunto(s) de datos|PSNR Rel.|SSIM Rel.|SAM|FID|
|----------|----------|--------|-------|----|-------|
|Red de reconstrucción G_R [Baidu](https://pan.baidu.com/s/10udDJaswX44SXg11a-Vn8Q?pwd=1234)/[Google](https://drive.google.com/file/d/13nKYFD2o7tyoV_K92hrMb11x0HayI4mD/view?usp=share_link)|MSAW|29.9690|0.7043|1.2850|70.8376|
|Red de traducción G_T [Baidu](https://pan.baidu.com/s/1Flie7J8K04oWeu0k8zCpqw?pwd=1234)/[Google](https://drive.google.com/file/d/1PcGqallpb8VOw3cXhUqNXg1WjAaB3QUl/view?usp=share_link)|MSAW|22.4651|0.3871|2.7256|102.1934|

* **El código para todos los enlaces de descarga de Baidu es `1234`**
## Inicio rápido
Para predecir la imagen óptica a partir de las imágenes correspondientes, descargue primero el modelo preentrenado de la columna `Model Name` en la tabla anterior. Algunas imágenes de prueba del conjunto de datos SpaceNet (conjunto de datos MSAW) se han incluido en el directorio "./dataset/SpaceNet/". Después de descargar el modelo de traducción o reconstrucción en el directorio: "./checkpoints/trans/latest_net_G_trans.pth" o "./checkpoints/recon/latest_net_G_recon.pth", puede probar los modelos con:
```
CUDA_VISIBLE_DEVICES=0 python\
 test.py\
 --name trans <nombre del experimento. Decide dónde almacenar muestras y modelos\
 --net translation <qué red desea probar (reconstrucción o traducción)>\
 ```
## Preparación de datos
El conjunto de datos MSAW se puede descargar con
```
aws s3 ls s3://spacenet-dataset/spacenet/SN6_buildings/
```
Y el conjunto de datos QXS-SAROPT y SEN1-2 se puede descargar mediante el [link1](https://github.com/yaoxu008/QXS-SAROPT) y el [link2](https://mediatum.ub.tum.de/1436631)
#### Establecer la ruta de datos
Cambie el "--dataroot" del archivo base_options.py a la ruta de su conjunto de datos.
La carpeta de cada conjunto de datos debe organizarse así (A para imagen óptica, B para imagen SAR):
```
<raíz del conjunto de datos>
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

Para diferentes conjuntos de datos, debe establecer "--input_nc" de manera diferente. Para el conjunto de datos MSAW con imágenes SAR de 4 canales: "--input_nc 4", para otros conjuntos de datos con 3 canales: "--input_nc 3". Y en "data/aligned_datasets.py" el método de carga de la imagen SAR debe cambiarse según el canal SAR.
## Entrenamiento y prueba
Puede usar los siguientes comandos para entrenar Parallal-Gan en SpaceNet. Como se menciona en nuestro artículo, entrenamos la red de reconstrucción y la red de traducción por separado. El entrenamiento se divide en dos etapas:
```
# entrenar la red de reconstrucción en la etapa1
CUDA_VISIBLE_DEVICES=0 python\
 train.py\
 --name recon\
 --net reconstruction
# y probar la red de reconstrucción:
 CUDA_VISIBLE_DEVICES=0 python\
 test.py\
 --name recon\
 --net reconstruction
```
Después de completar la etapa 1, copie la ruta del último modelo guardado `./checkpoints/name/latest_net_G_recon.pth` en `load_path` de parallel_gan_model.py, y use el siguiente comando para continuar el entrenamiento:
```
# entrenar la red de traducción en la etapa2
CUDA_VISIBLE_DEVICES=0 python\
 train.py\
 --name trans\
 --net translation
# y probar la red de traducción:
 CUDA_VISIBLE_DEVICES=0 python\
 train.py\
 --name trans\
 --net translation
 ```

## Reconocimientos
Parte de este repositorio proviene de [Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

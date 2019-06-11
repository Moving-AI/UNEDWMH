# Aumento de datos para imagen médica mediante el uso de GAN

En este código se implementa una WGAN-GP con un generador DCGAN. 

## Prerrequisitos

```
Keras==2.2.4
tensorflow_gpu==1.13.1
numpy==1.16.3
matplotlib==3.0.3
opencv_python==4.1.0.25
Pillow==6.0.0
protobuf==3.8.0
tensorflow==1.13.1
```

Con Python 3.6

## Instalación
1. Instalar Python 3.6
2. Instalar todas las dependencias
3. Clona/descarga este repositorio
4. Añade los datasets ```images_three_datasets_sorted.npy``` y ```masks_three_datasets_sorted.npy``` a la carpeta raíz (no incluidos en este repositorio)
5. Lanza ```get_images.ipynb``` para reducir la muestra de imágenes
6. Modifica los hiperparámetros de las GAN en ```wgan_tensorboard.ipynb```
7. Lanza ```wgan_tensorboard.ipynb``` 

## Implementación
Se ha realizado enteramente sobre Python, y consta de los siguientes archivos:
* ```get_images.ipynb```: reduce el tamaño del conjunto de datos, tomando las slices centrales de cada paciente. Requiere los archivos ```images_three_datasets_sorted.npy``` y ```masks_three_datasets_sorted.npy```, que no están incluidos en el repositorio.
* ```wgan_tensorboard.ipynb```: Programa principal que se encarga de entrenar las redes. En la segunda celda se pueden modificar algunos de los parámetros del programa
* ```funciones_wgan.py```: compendio de funciones útiles para la WGAN

A lo largo de los notebook pueden encontrarse explicaciones de los pasos realizados en cada punto.

## Agradecimientos
Agradecemos a Hongwei Li et al. por proveer el dataset: https://github.com/hongweilibran/wmh_ibbmTum

La implementación de WGAN-GP se ha modificado partiendo de https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

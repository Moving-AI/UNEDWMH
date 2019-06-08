# Aumento de datos para imagen médica mediante el uso de GAN

En este código se implementa una WGAN con un generador DCGAN. El código está estructurado de la siguiente manera:

El notebook get_images.ipynb realiza una selección de las imágenes originales para entrenar la GAN más rápido y con mejores resultados. Para cada paciente coge las slices centrales. Una vez se ejecuta se obtiene como resultado el archivo muestra_seleccionada.ipynb, que será utilizado por wgan_tensorboard.ipynb

El resto del código está en el notebook wgan_tensorboard.ipynb. Los paquetes necesarios para ejecutar el código son:

numpy
keras
matplotlib.pyplot
functools
os
PIL (pillow)
cv2
time
tensorflow
datetime

A lo largo del notebook pueden encontrarse explicaciones de los pasos realizados en cada punto.
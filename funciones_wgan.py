"""An implementation of the improved WGAN described in https://arxiv.org/abs/1704.00028
The improved WGAN has a term in the loss function which penalizes the network if its
gradient norm moves away from 1. This is included because the Earth Mover (EM) distance
used in WGANs is only easy to calculate for 1-Lipschitz functions (i.e. functions where
the gradient norm has a constant upper bound of 1).
The original WGAN paper enforced this by clipping weights to very small values
[-0.01, 0.01]. However, this drastically reduced network capacity. Penalizing the
gradient norm is more natural, but this requires second-order gradients. These are not
supported for some tensorflow ops (particularly MaxPool and AveragePool) in the current
release (1.0.x), but they are supported in the current nightly builds
(1.1.0-rc1 and higher).
To avoid this, this model uses strided convolutions instead of Average/Maxpooling for
downsampling. If you wish to use pooling operations in your discriminator, please ensure
you update Tensorflow to 1.1.0-rc1 or higher. I haven't tested this with Theano at all.
The model saves images using pillow. If you don't have pillow, either install it or
remove the calls to generate_images.
"""

import numpy as np
from keras.layers.merge import _Merge
from keras import backend as K
import matplotlib.pyplot as plt
from functools import partial
import os

try:
    from PIL import Image
except ImportError:
    print('This script depends on pillow! Please install it (e.g. with pip install pillow)')

BATCH_SIZE = 8
# The training ratio is the number of discriminator updates
# per generator update. The paper uses 5.
TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for
    display."""
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def generate_images(generator_model, output_dir, epoch, n_images, method='FLAIR'):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG
    file."""
    images = generator_model.predict(np.random.rand(n_images, 128))
    cols = 2

    fig = plt.figure()
    for n, image in enumerate(images):
        a = fig.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
        if method == 'T1':
            plt.imshow(image[..., 0], cmap='gray')
        else:
            plt.imshow(image[..., 1], cmap='gray')

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.savefig(f'{output_dir}{method}/epoch_{epoch}.png')
    plt.close(fig)

def sample_best_images(generator_model, discriminator, output_dir, epoch='No', n_images = 10, n_images_total = 100, save = True):
    """Coger las mejores imágenes.
    Cogemos por defecto 100 imágenes del generador (controlándolo con n_images_total)
    Seleccionamos las n_images mejores y las guardamos en un archivo. Guardamos por separado las
    FLAIR y las T1.
    """
    images = generator_model.predict(np.random.rand(n_images_total, 128))
    images_mark = discriminator.predict(images).reshape((n_images_total))
    order = np.argsort(-images_mark)[:n_images]

    cols = 2
    images_final = images[order,...]
    print(images.shape)

    
    if save:
        figFLAIR = plt.figure()
        for n, image in enumerate(images_final):
            a = figFLAIR.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
            plt.imshow(image[..., 1], cmap='gray')
        if not os.path.isdir(f'{output_dir}FLAIR/'):
            os.mkdir(f'{output_dir}FLAIR/')

        figFLAIR.set_size_inches(np.array(figFLAIR.get_size_inches()) * n_images)
        figFLAIR.savefig(f'{output_dir}FLAIR/epoch_{epoch}.png')
        plt.close(figFLAIR)
    
        figT1 = plt.figure()
        for n, image in enumerate(images_final):
            a = figT1.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
            plt.imshow(image[..., 0], cmap='gray')
        
        if not os.path.isdir(f'{output_dir}T1/'):
            os.mkdir(f'{output_dir}T1/')
        
        figT1.set_size_inches(np.array(figT1.get_size_inches()) * n_images)
        print(f'{output_dir}T1/epoch_{epoch}.png')
        figT1.savefig(f'{output_dir}T1/epoch_{epoch}.png')
        plt.close(figT1)

        figMask = plt.figure()
        for n, image in enumerate(images_final):
            a = figMask.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
            plt.imshow(image[..., 2], cmap='gray')
        
        if not os.path.isdir(f'{output_dir}Mask/'):
            os.mkdir(f'{output_dir}Mask/')
        
        figMask.set_size_inches(np.array(figMask.get_size_inches()) * n_images)
        print(f'{output_dir}Mask/epoch_{epoch}.png')
        figMask.savefig(f'{output_dir}Mask/epoch_{epoch}.png')
        plt.close(figMask)
    else:
        fig = np.empty((3,256, 256,1))
        fig[0,...] = images_final[...,0].reshape((1 ,256, 256, 1))
        fig[1,...] = images_final[...,1].reshape((1 ,256, 256, 1))
        fig[2,...] = images_final[...,2].reshape((1 ,256, 256, 1))

        return fig
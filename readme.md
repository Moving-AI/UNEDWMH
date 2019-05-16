# Introduction

En el código

Keras implementation of DCGAN and WGAN for image augmentation.

# Datasets

- Utretch, Singapore, GE3T:
  - pre: Contains the original FLAIR and T1 images used in the code, masks and image processings. Can be used as extra channels in the training input.
    - Flair_enhanced: FLAIR images with improved contrast in WM
    - distWMBorder_Danielsson & Maurer: WM border distance maps
  - wmh.nii.gz: gold standard segmentation.
- images_three_datasets_sorted, masks_three_datasets_sorted: images and masks already preprocessed with the steps described in Utrecht_preprocessing, GE3T_preprocessing.

# Scripts

- train_leave_one_out & test_leave_one_out.py: train and test U-Net model with leave-one-out protocol.
- evaluation.py: segmentation evaluation metrics

[Model description](https://www.sciencedirect.com/science/article/pii/S1053811918305974?via%3Dihub)
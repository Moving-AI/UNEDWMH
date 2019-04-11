DATASETS:

    - Utrecht, Singapore, GE3T:
        - pre: contiene las imágenes FLAIR y T1 originales (2D) utilizadas en el código, máscaras y procesados de las imágenes (por ejemplo, pueden ser utilizados como canales extra en el input para el entrenamiento)
            - FLAIR_enhanced: FLAIR con mejora del contraste en WM.
            - distWMborder_Danielsson y Maurer: mapas de distancias al borde de WM.
        - wmh.nii.gz: segmentación gold standard.
        
    - images_three_datasets_sorted, masks_three_datasets_sorted: imágenes y máscaras ya preprocesadas con los pasos descritos en Utrecht_preprocessing, GE3T_preprocessing (tamaño original de las imágenes reducido o aumentado a 200x200 y  eliminar primeras y últimas slices de cada imagen - Utrecht, Singapore pasan de 48 a 38 slices, GE3T de 83 a 63).

SCRIPTS:

    - train_leave_one_out.py y test_leave_one_out.py: train y test U-Net models con el protocolo leave-one-out.
    - evaluation.py: métricas de evaluación de la segmentación.


Descripción del método: https://www.sciencedirect.com/science/article/pii/S1053811918305974?via%3Dihub

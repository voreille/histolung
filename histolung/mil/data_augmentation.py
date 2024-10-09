# Code reproduced from https://github.com/lluisb3/histo_lung/blob/main/training/train_MIL_k_fold.py

import numpy as np
import albumentations as A


def select_parameters_colour():
    hue_min = -15
    hue_max = 8

    sat_min = -20
    sat_max = 10

    val_min = -8
    val_max = 8

    p1 = np.random.uniform(hue_min, hue_max, 1)
    p2 = np.random.uniform(sat_min, sat_max, 1)
    p3 = np.random.uniform(val_min, val_max, 1)

    return p1[0], p2[0], p3[0]


def select_rgb_shift():
    r_min = -10
    r_max = 10

    g_min = -10
    g_max = 10

    b_min = -10
    b_max = 10

    p1 = np.random.uniform(r_min, r_max, 1)
    p2 = np.random.uniform(g_min, g_max, 1)
    p3 = np.random.uniform(b_min, b_max, 1)

    return p1[0], p2[0], p3[0]


def select_elastic_distorsion():
    sigma_min = 0
    sigma_max = 20

    alpha_affine_min = -20
    alpha_affine_max = 20

    p1 = np.random.uniform(sigma_min, sigma_max, 1)
    p2 = np.random.uniform(alpha_affine_min, alpha_affine_max, 1)

    return p1[0], p2[0]


def select_scale_distorsion():
    scale_min = 0.8
    scale_max = 1.0

    p1 = np.random.uniform(scale_min, scale_max, 1)

    return p1[0]


def select_grid_distorsion():
    dist_min = 0
    dist_max = 0.2

    p1 = np.random.uniform(dist_min, dist_max, 1)

    return p1[0]


def get_augmentations_pipeline(prob=0.5):
    list_operations = []
    probas = np.random.rand(7)

    if (probas[0] > prob):
        #print("VerticalFlip")
        list_operations.append(A.VerticalFlip(always_apply=True))
    if (probas[1] > prob):
        #print("HorizontalFlip")
        list_operations.append(A.HorizontalFlip(always_apply=True))
    #"""
    if (probas[2] > prob):
        #print("RandomRotate90")
        #list_operations.append(A.RandomRotate90(always_apply=True))

        p_rot = np.random.rand(1)[0]
        if (p_rot <= 0.33):
            lim_rot = 90
        elif (p_rot > 0.33 and p_rot <= 0.66):
            lim_rot = 180
        else:
            lim_rot = 270
        list_operations.append(
            A.SafeRotate(always_apply=True,
                         limit=(lim_rot, lim_rot + 1e-4),
                         interpolation=1,
                         border_mode=4))

    if (probas[3] > prob):
        #print("HueSaturationValue")
        p1, p2, p3 = select_parameters_colour()
        list_operations.append(
            A.HueSaturationValue(always_apply=True,
                                 hue_shift_limit=(p1, p1 + 1e-4),
                                 sat_shift_limit=(p2, p2 + 1e-4),
                                 val_shift_limit=(p3, p3 + 1e-4)))

    if (probas[4] > prob):
        p1 = select_scale_distorsion()
        list_operations.append(
            A.RandomResizedCrop(height=224,
                                width=224,
                                scale=(p1, p1 + 1e-4),
                                always_apply=True))
        #print(p1,p2,p3)

    if (probas[5] > prob):
        #p1, p2 = select_elastic_distorsion()
        list_operations.append(
            A.ElasticTransform(alpha=1,
                               border_mode=4,
                               sigma=5,
                            #    alpha_affine=5, # is not in newer version?
                               always_apply=True))
        #print(p1,p2)

    if (probas[6] > prob):
        p1 = select_grid_distorsion()
        list_operations.append(
            A.GridDistortion(num_steps=3,
                             distort_limit=p1,
                             interpolation=1,
                             border_mode=4,
                             always_apply=True))
        #print(p1)

    return A.Compose(list_operations)

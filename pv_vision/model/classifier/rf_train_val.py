import numpy as np
import cv2 as cv

from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from sklearn import utils as sk_utils
from sklearn.metrics import f1_score

import itertools

import copy



def im_flatten(images):
    """Flatten the grayscale image

    Parameters
    ----------
    images: array
    The array of grayscale images

    Returns
    -------
    flattened_images: array
    Matrix, row of which is a flattened image
    """
    return images.reshape(len(images), -1)


def im_aug(images, labels, shuffle=True, select_aug=False, select_inx=None):
    """Do data augmentation, including vertical flip, horizontal flip and 180 rotation

    Parameters
    ----------
    images: array
    Array of grayscale images

    labels: list or array
    Labels of the images

    shuffle: bool
    If true, the augmented images will be shuffled

    select_aug: bool
    If true, only the images with labels in select_inx will be argumented

    select_inx: list
    Labels of the images to be augmented

    Returns
    -------
    x: array
    Array of augmented images

    y: array
    Array of labels
    """
    v_flip = []
    h_flip = []
    rotate = []
    labels_few = []

    if not select_aug:
        for image in images:
            v_flip.append(cv.flip(image, 0))
            h_flip.append(cv.flip(image, 1))
            rotate.append(cv.rotate(image, cv.ROTATE_180))
    elif select_aug:
        for i, image in enumerate(images):
            if labels[i] in select_inx:
                v_flip.append(cv.flip(image, 0))
                h_flip.append(cv.flip(image, 1))
                rotate.append(cv.rotate(image, cv.ROTATE_180))
                labels_few.append(labels[i])

    v_flip = np.array(v_flip)
    h_flip = np.array(h_flip)
    rotate = np.array(rotate)
    if not select_aug:
        x, y = np.concatenate((images, v_flip, h_flip, rotate)), np.concatenate((labels, labels, labels, labels))
    elif select_aug:
        labels_few = np.array(labels_few)
        x, y = np.concatenate((images, v_flip, h_flip, rotate)), np.concatenate(
            (labels, labels_few, labels_few, labels_few))

    if shuffle:
        return sk_utils.shuffle(x, y)
    else:
        return x, y


def iterate_values(S):
    """An iteration tool to return combined hyperparameters

    Parameters
    ----------
    S: dict
    The dictionary of hyperparameters

    Returns
    -------
    Combined hyperparameters: generator
    The generator of combined hyperparameters dict
    """
    keys, values = zip(*S.items())

    for row in itertools.product(*values):
        yield dict(zip(keys, row))


def train_val_rf(X_train, y_train, X_val, y_val, rf_para):
    """Train and validate random forest. Use macro F1 score to validate

    Parameters
    ----------
    X_train, X_val: array
    Matrix of training X, validation X.

    y_train, y_val: array
    Array of training y, validation y

    rf_para: dict
    Dictionary of random forest parameters

    Returns
    -------
    rf: RandomForestClassifier class
    Trained model

    f1_macro: float
    Macro F1 score on validation set
    """
    rf = RandomForestClassifier(**rf_para, n_jobs=-1)
    rf.fit(X_train, y_train)

    pred_val = rf.predict(X_val)
    f1_macro = f1_score(y_val, pred_val, average='macro')

    return rf, f1_macro


def random_search_rf(images_train, y_train, images_val, y_val, iteration, para_grid):
    """Hyperparameter tuning using random search

    Parameters
    ----------
    images_train, images_val: array
    Array of training images, validation images

    y_train, y_val: array
    Array of training labels, validation labels

    iteration: int
    Iteration of hyperparameter tuning, i.e., number of combination of hyperparameters

    para_grid: dict
    Dictionary of hyperparameters
    e.g. rf_para_grid = {
            'n_estimators': [10, 20, 30, 40, 50, 80, 100, 200, 400, 1000],
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 20, 40, 60, 80, 100, None],
            'bootstrap': [True, False]
        }

    Returns
    -------
    best_model: RandomForestClassifier class
    Trained model of optimal macro F1 score on validation set

    best_rf_para: dict
    Dictionary of parameters of model of optimal macro F1 score on validation set

    best_score: float
    Optimal macro F1 score on validation set
    """
    X_train = im_flatten(images_train)
    X_val = im_flatten(images_val)

    rf_para_list = list(iterate_values(para_grid))
    np.random.seed(42)
    rf_para_selected = np.random.choice(rf_para_list, iteration, replace=False)

    best_rf_para = {}
    best_score = 0
    for rf_para in tqdm(rf_para_selected):
        model, current_score = train_val_rf(X_train, y_train, X_val, y_val, rf_para)
        if current_score >= best_score:
            best_score = current_score
            best_rf_para = rf_para
            best_model = copy.deepcopy(model)
    return best_model, best_rf_para, best_score


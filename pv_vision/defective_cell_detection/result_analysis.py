import os.path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score
import pickle
from pathlib import Path
from sklearn import metrics
import seaborn as sns
import pandas as pd
import json


def predict_failed(y_test, pred_test, images_test, model_name=None, save_path=None):
    """Output failed predictions

    Parameters
    ----------
    y_test: list or array
    True label of testing set

    pred_test: list or array
    Predicted label of testing set

    images_test: list or array
    Images in testing set

    model_name: str
    Name of machine learning model

    save_path: str or pathlib.PosixPath
    Path to store the output

    Returns
    -------
    failed_test: dict
    Dictionary that contains the images, true labels and predicted labels of failures
    """
    failed_test = {
        'image': [],
        'truth': [],
        'pred': []
    }

    for i, truth in enumerate(y_test):
        if truth != pred_test[i]:
            failed_test['image'].append(images_test[i])
            failed_test['truth'].append(truth)
            failed_test['pred'].append(pred_test[i])
    if save_path:
        with open(Path(save_path) / (model_name + '_failed.pkl'), 'wb') as f:
            pickle.dump(failed_test, f)

    return failed_test


def add_fbeta(report, y_true, y_pred, fbeta=2):
    """Add fbeta metric into sklearn.metrics.classification_report

    Parameters
    ----------
    report: pandas.core.frame.DataFrame
    Dataframe from sklearn.metrics.classification_report

    y_true: list or array
    True label of testing set

    y_pred: list or array
    Predicted label of testing set

    fbeta: int
    Beta value in f-beta score

    Returns
    -------
    report: pandas.core.frame.DataFrame
    Report that contains fbeta score
    """
    f2_score = list(fbeta_score(y_true, y_pred, average=None, beta=fbeta))
    f2_score.append(fbeta_score(y_true, y_pred, average='micro', beta=fbeta))
    f2_score.append(fbeta_score(y_true, y_pred, average='macro', beta=fbeta))
    f2_score.append(fbeta_score(y_true, y_pred, average='weighted', beta=fbeta))

    report.loc[f'f{fbeta}-score'] = f2_score

    return report


def metrics_report(y_test, pred_test, model_name=None, save_path=None, fbeta=2, le=None, label_names=None):
    """Output the metrics report using sklearn.metrics.classification_report

    Parameters
    ----------
    y_test: list or array
    True label of testing set

    pred_test: list or array
    Predicted label of testing set

    model_name: str
    Name of machine learning model

    save_path: str or pathlib.PosixPath
    Path to store the output

    fbeta: int
    Beta value of F-beta score. Default is 2

    le: sklearn.preprocessing.LabelEncoder
    Label encoder. Otherwise need to configure label_names

    label_names: list
    Names of labels. Each label name's index means y number

    Returns
    -------
    test_report: pandas.core.frame.DataFrame
    Metrics report
    """
    if not label_names:
        label_names = le.classes_
    test_report = pd.DataFrame(metrics.classification_report(y_test, pred_test,
                                                             target_names=label_names, output_dict=True))
    test_report = add_fbeta(test_report, y_test, pred_test, fbeta)

    if save_path:
        test_report.to_pickle(save_path / (model_name + '_report.pkl'))

    return test_report


def draw_cm(defect_name, solar_df=None, y_true=None, y_pred=None, diagonal=True, cbar_size=0.6, linewidth=0.5, save_path=None, model_name=None):
    """Plot the confusion matrix

    Parameters
    ----------
    solar_df: dict or pandas.core.frame.DataFrame
    Dataframe that indicates the true label and predicted label of all solar cells. Column names for true label and
    predicted should be 'y_true' and 'y_pred'.
    If None, y_true and y_pred must be passed

    y_true, y_pred: array or list
    True label and predicted label of all solar cells. It should be passed if solar_df is not passed

    defect_name: dict
    Mapping the value of y into label names.
    E.g. defect_name = {
            0: 'crack',
            1: 'intact',
            2: 'intra',
            3: 'oxygen',
            4: 'solder'
        }

    diagonal: bool
    If true, the true positive number (diagonal of confusion matrix) is shown.

    cbar_size: float
    Size of color bar. Default is 0.6. Check 'sns.heatmap' for more information.

    linewidth: float
    Line width of confusion matrix. Default is 0.5. Check 'sns.heatmap' for more information.

    model_name: str
    Name of machine learning model

    save_path: str or pathlib.PosixPath
    Path to store the output

    Returns
    -------
    solar_pv: pivot
    Pivot table of confusion matrix. Index is ground truth and column is prediction.

    """
    if not solar_df:
        solar_pv = pd.DataFrame({"y_true": y_true, "y_pred":y_pred}).groupby(['y_true', 'y_pred'], as_index=False).size().pivot(
            index='y_true', columns='y_pred', values='size').rename(columns=defect_name, index=defect_name).fillna(0)
    else:
        solar_pv = pd.DataFrame(solar_df)[['y_true', 'y_pred']].groupby(['y_true', 'y_pred'], as_index=False).size().pivot(
            index='y_true', columns='y_pred', values='size').rename(columns=defect_name, index=defect_name).fillna(0)
    
    if not diagonal:
        np.fill_diagonal(solar_pv.values, np.nan)

    ax = sns.heatmap(data=solar_pv, annot=True, fmt='g', cmap='Blues', square=True, cbar_kws={'shrink': cbar_size},
                     linewidths=linewidth)
    plt.xlabel('Pred')
    plt.ylabel('Truth')

    if save_path:
        plt.tight_layout()
        plt.savefig(Path(save_path) / f'{model_name}_cm.png', bbox_inches=0, dpi=600)


def score_compare(reports_dic, score_type, category, hue_order, save_path=None, output=None,
                  legend_pos=(0.5, 0.4), legend_col=2, label_space=1):
    """Compare the metric of different models.

    reports_dic: dict
    Dictionary of metrics report of different models. Metrics report can be output by 'result_analysis.metrics_report'.

    score_type: str
    Type of metric to be compared, e.g. 'recall', 'f1-score'.

    category: list
    List of categories to be compared, e.g. ['crack', 'intact', 'intra', 'oxygen', 'solder']

    hue_order: list
    List of models to be compared.

    save_path: str or pathlib.PosixPath
    Path to store the output

    output: Bool
    If True, return the melted dataframe

    legend_pos, legend_col, label_space:
    Property of the legend. Check matplotlib.pyplot.legend for more information
    """
    scores = pd.DataFrame()
    for model in reports_dic:
        scores[model] = reports_dic[model].loc[score_type, category]

    melt = scores.reset_index().melt('index', var_name='model', value_name=score_type)
    ax = sns.barplot(data=melt, x='index', y=score_type, hue='model', hue_order=hue_order)
    # ax.set_title('Comparison of '+score_type.title())
    ax.set_xlabel('Category')
    ax.set_ylabel(score_type.title())
    plt.legend(loc='upper center', bbox_to_anchor=legend_pos, ncol=legend_col, columnspacing=label_space)
    plt.tight_layout()
    if save_path:
        plt.savefig(Path(save_path) / f'comparison_{score_type}.png', dpi=600)
    if output:
        return melt


# Especially used to collect the labels from YOLO model
def coordinate2inx(coordinate, row=8, col=16, im_shape=[300, 600]):
    """Convert coordinate of top-left corner of bbox into index.
    Index on solar module looks like:
    [[0, 1, 2]
     [3, 4, 5]]

     Parameters
     ----------
     coordinate: list
     [x, y] of top-left corner of bbox

     row, col: int
     number of rows and columns of solar module

     im_shape: list
     Shape of the module image in the form of [height, width]

     Returns
     -------
     inx: int
     Index of the bbox
    """
    inx = col * round(coordinate[1] / (im_shape[0] / row)) + round(coordinate[0] / (im_shape[1] / col))

    return inx


def coordinate2coordinate(coordinate, row=8, col=16, im_shape=[300, 600]):
    """Convert coordinate of top-left corner of bbox into coordinate of solar cell (x_col, y_row).
    Coordinate of solar cell on solar module looks like:
    [[(0,0), (1,0), (2,0)]
     [(0,1), (1,1), (2,1)]]

     Parameters
     ----------
     coordinate: list
     [x, y] of top-left corner of bbox

     row, col: int
     number of rows and columns of solar module

     im_shape: list
     Shape of the module image in the form of [height, width]

     Returns
     -------
     coordinate_cell: int
     (x_col, y_row) of the bbox
    """
    return round(coordinate[0] / (im_shape[1] / col)), round(coordinate[1] / (im_shape[0] / row))


def inx2coordinate(inx, row=8, col=16):
    """Convert inx of bbox into coordinate of solar cell (x_col, y_row).
    Coordinate of solar cell on solar module looks like:
    [[(0,0), (1,0), (2,0)]
     [(0,1), (1,1), (2,1)]]

     Parameters
     ----------
     inx: int
     Index of the bbox

     row, col: int
     number of rows and columns of solar module

     Returns
     -------
     coordinate_cell: int
     (x_col, y_row) of the bbox
    """
    return inx % col, inx // col


def collect_all_cells(ann_path, labels_pred=None, labels_true=None, pred_key="labels_pred", true_key="labels_true", row_col=[8, 16], shape=[300, 600]):
    """Collect the information of all the cells in one module processed by YOLO model

    Parameters
    ----------
    ann_path: str or pathlib.PosixPath
    Path of ann files

    labels_pred, labels_true: dict
    Mapping label names into names of defects.
    E.g. true label:
    label2defect = {
    "crack_bbox": "crack",
    "oxygen_bbox": "oxygen",
    "solder_bbox": "solder",
    "intra_bbox": "intra"
    }

    pred_key, true_key: str
    Key name of labels in the output "cell_info"

    row_col: list
    Number of rows/columns of module

    shape: list
    Shape of the module image in the form of [height, width]

    Returns
    -------
    cell_info: dict
    The dict that contains information of all the cells on one module.
    Information includes name of the module, index of the cell, labels of prediction/truth, x_col and y_row of the cell
    """
    module_name = os.path.splitext(os.path.split(ann_path)[-1])[0]
    #module_name = str(ann_path).split('/')[-1].split('.')[0]

    all_names = np.full(row_col[0] * row_col[1], fill_value=module_name).tolist()
    all_index = list(range(row_col[0] * row_col[1]))
    all_x = np.tile(list(range(row_col[1])), row_col[0]).tolist()
    all_y = np.tile(list(range(row_col[0])), (row_col[1], 1)).T.flatten().tolist()

    cell_info = {
        'module_name': all_names,
        'index': all_index,
        'x': all_x,
        'y': all_y,
    }

    with open(ann_path, 'r') as file:
        data = json.load(file)
    if labels_true:
        all_labels = np.full(row_col[0] * row_col[1], fill_value='intact').tolist()
        if len(data['objects']) > 0:
            for obj in data['objects']:
                classTitle = obj['classTitle']
                if classTitle in labels_true.keys():
                    points = obj['points']['exterior'][0]
                    inx = coordinate2inx(points, row=row_col[0], col=row_col[1], im_shape=shape)
                    all_labels[inx] = labels_true[classTitle]
        cell_info[true_key] = all_labels
    if labels_pred:
        all_second_labels = np.full(row_col[0] * row_col[1], fill_value='intact').tolist()
        if len(data['objects']) > 0:
            for obj in data['objects']:
                classTitle = obj['classTitle']
                if classTitle in labels_pred.keys():
                    points = obj['points']['exterior'][0]
                    inx = coordinate2inx(points, row=row_col[0], col=row_col[1], im_shape=shape)
                    all_second_labels[inx] = labels_pred[classTitle]
        cell_info[pred_key] = all_second_labels
    return cell_info


def collect_defects(ann_path, defects_dic, labels, row_col=[8, 16], shape=[300, 600], mode=1):
    """Only collect the information of defective cells.
    This method is used when applying YOLO model on data without true labels

    Parameters
    ----------
    ann_path: str or pathlib.PosixPath
    Path of ann files

    defects_dic: dict
    E.g.
    defects_info = {
        'module_name': [],
        'index': [],
        'defects': [],
        'x': [],
        'y': [],
        'confidence': []
        }

    labels: dict
    Mapping label names into names of defects.
    E.g. true label:
    label2defect = {
    "crack_bbox": "crack",
    "oxygen_bbox": "oxygen",
    "solder_bbox": "solder",
    "intra_bbox": "intra"
    }

    row_col: list
    Number of rows/columns of module

    shape: list
    Shape of the module image in the form of [height, width]

    mode: int
    If 0, collect the true labels
    If 1, collect the predicted labels

    Returns
    -------
    defects_dic: dict
    The dict that contains information of all defective cells on one module.
    Information includes name of the module, index of the cell, labels of prediction, x_col and y_row of the cell,
    confidence of the model
    """

    module_name = os.path.splitext(os.path.split(ann_path)[-1])[0]
    #module_name = str(ann_path).split('/')[-1].split('.')[0]
    if module_name in defects_dic['module_name']:
        return defects_dic

    with open(ann_path, 'r') as file:
        data = json.load(file)
    if len(data['objects']) > 0:
        for obj in data['objects']:
            classTitle = obj['classTitle']
            points = obj['points']['exterior'][0]
            inx = coordinate2inx(points, row=row_col[0], col=row_col[1], im_shape=shape)
            # x, y = coordinate2coordinate(points, row=row_col[0], col=row_col[1], im_shape=shape)
            x, y = inx2coordinate(inx, row=row_col[0], col=row_col[1])
            defects_dic['module_name'].append(module_name)
            defects_dic['index'].append(inx)
            defects_dic['defects'].append(labels[classTitle])
            defects_dic['x'].append(x)
            defects_dic['y'].append(y)
            if mode == 1:
                confidence = obj['tags'][0]['value']
                defects_dic['confidence'].append(confidence)

    return defects_dic


def get_label_one_module(ann_path, defects_dic, row_col=[8, 16], shape=[300, 600], fill_label=1):
    """Only collect the label(digit label) information of cells in one module

    Parameters
    ----------
    ann_path: str or pathlib.PosixPath
    Path of ann files

    defects_dic: dict
    Mapping the name of labels into digit
    e.g.
    defects_dic_yolo = {
        'crack_bbox_yolo': 0,
        'solder_bbox_yolo': 4,
        'intra_bbox_yolo': 2,
        'oxygen_bbox_yolo': 3

    }

    row_col: list
    Number of rows/columns of module

    shape: list
    Shape of the module image in the form of [height, width]

    fill_label: int
    The digit label of intact cells. Default is 1

    Returns
    -------
    yolo_labels: list
    Digit labels of cells in one module. Index of the list is the index of the cell in the module
    """
    yolo_labels = np.full(row_col[0] * row_col[1], fill_value=fill_label)
    with open(ann_path, 'r') as file:
        data = json.load(file)
    if len(data['objects']) > 0:
        for obj in data['objects']:
            classTitle = obj['classTitle']
            if classTitle in defects_dic:
                points = obj['points']['exterior'][0]
                inx = coordinate2inx(points, row=row_col[0], col=row_col[1], im_shape=shape)
                yolo_labels[inx] = defects_dic[classTitle]

    return yolo_labels


# Check if the module image needs rotation such that the longer edge is closer to the ground
def need_rotate(corners):
    """Check if need rotation. If the left edge is longer than the right one, then rotate.

    Parameters
    ----------
    corners: array
    Sorted coordinates of module corners. The order is top-left, top-right, bottom-left and bottom-right

    Returns
    -------
    bool
    """
    if np.sum((corners[1] - corners[3]) ** 2) < np.sum((corners[0] - corners[2]) ** 2):
        return True
    else:
        return False


def rotate_points(points, width=600, height=300):
    """180 degree rotation of points of bbox

    Parameters
    ----------
    points: list or array
    Coordinates of top-left, top-right, bottom-left and bottom-right points

    width, height: int
    Width/height of perspective transformed module image

    Returns
    -------
    rotated_points: list or array
    180 degree rotation
    """
    return [[width - points[1][0], height - points[1][1]], [width - points[0][0], height - points[0][1]]]


def rotate_ann(origin_dir, ann, save_dir):
    """Rotate the bbox in ann files

    Parameters
    ----------
    origin_dir: str or pathlib.PosixPath
    Directory of original YOLO ann files

    ann: str
    Name of the ann file

    save_dir:
    str or pathlib.PosixPath
    Directory to store rotated YOLO ann files
    """
    with open(Path(origin_dir) / ann, 'r') as file:
        data = json.load(file)
        size = data['size']
    if len(data['objects']) > 0:
        for obj in data['objects']:
            # classTitle = obj['classTitle']
            points = obj['points']['exterior']
            obj['points']['exterior'] = rotate_points(points, width=size['width'], height=size['height'])

    with open(Path(save_dir) / ann, 'w') as file:
        json.dump(data, file)


# position distribution
def plot_heatmap(dataframe, category, defect_col='defects', bar_color='Blues', linewidths=0.5, cbar_size=0.8):
    """Plot the distribution of defective cells

    Parameters
    ----------
    dataframe: Pandas dataframe
    The dataframe of defects.
    Normally use pd.DataFrame(defects_dict) where defects_dict is obtained from "collect_defects()"

    category: str
    The name of the defect to plot, e.g. "crack"

    defect_col: str
    The name of column of the defects

    bar_color: str
    Color of heatmap

    linewidths: float
    Width of separation lines in the heatmap

    cbar_size: float
    Size of colorbar

    Returns
    -------
    defect_pv: pivot table
    pivot table of defect distribution
    """
    defect_df = dataframe.loc[dataframe[defect_col] == category, ['x', 'y']]
    defect_pv = defect_df.groupby(['x', 'y'], as_index=False).size().pivot('y', 'x', 'size')

    ax = sns.heatmap(defect_pv, square=True, cbar_kws={'shrink':cbar_size}, cmap=bar_color, linewidths=linewidths)
    #ax.invert_yaxis()
    ax.xaxis.set_ticks_position('top')
    #ax.set_title(category.split('_')[0].title())

    return defect_pv

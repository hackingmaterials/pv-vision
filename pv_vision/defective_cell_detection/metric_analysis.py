import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score
import pickle
from pathlib import Path
from sklearn import metrics
import seaborn as sns
import pandas as pd


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


from shared_util.common import *
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score


def evaluate_ftc(y_pred, y_true, dataset_type, ent_types):
    evaluate_result = dict()

    logger.info('----------')
    logger.info(f'fault type classification evaluation dataset type: {dataset_type}')
    for ent_type in ent_types:
        ent_y_pred = np.array(y_pred[ent_type])
        ent_y_true = np.array(y_true[ent_type])
        fc_result = generate_ftc_metrics(ent_y_pred, ent_y_true)
        convert = {
            'p': 'precision',
            'r': 'recall',
            'f1': 'f1'
        }
        for em in ['p', 'r', 'f1']:
            logger.info(f'{ent_type.ljust(8) + convert[em].ljust(9)} | micro: {fc_result["micro_" + convert[em] + "_score"]:.4f}; macro: {fc_result["macro_" + convert[em] + "_score"]:.4f}; score: {fc_result[convert[em] + "_score"]}')
        evaluate_result[ent_type] = fc_result
    logger.info('----------')
    return evaluate_result


def generate_ftc_metrics(y_pred, y_true):
    evaluation_metric_dict = dict()

    evaluation_metric_dict['confusion_matrix'] = multilabel_confusion_matrix(y_true, y_pred)
    for i in ['micro', 'macro']:
        for j in ['precision_score', 'recall_score', 'f1_score']:
            evaluation_metric_dict[f'{i}_{j}'] = eval(f'{j}(y_true, y_pred, average="{i}", zero_division=0)')

    for j in ['precision_score', 'recall_score', 'f1_score']:
        evaluation_metric_dict[f'{j}'] = eval(f'{j}(y_true, y_pred, average=None, zero_division=0)')

    return evaluation_metric_dict

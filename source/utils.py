# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s
import time
from scipy import stats
import numpy as np


def c_index(y_true, y_pred):
    """
    Concordance Index Function

    Args:
    - y_trues: true values
    - y_pred: predicted values

    """

    matrix_pred = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    matrix_pred = tf.cast(matrix_pred == 0.0, tf.float32) * 0.5 + tf.cast(matrix_pred > 0.0, tf.float32)

    matrix_true = tf.subtract(tf.expand_dims(y_true, -1), y_true)
    matrix_true = tf.cast(matrix_true > 0.0, tf.float32)

    matrix_true_position = tf.where(tf.equal(matrix_true, 1))

    matrix_pred_values = tf.gather_nd(matrix_pred, matrix_true_position)

    # If equal to zero then it returns zero, else return the result of the division
    result = tf.where(tf.equal(tf.reduce_sum(matrix_pred_values), 0), 0.0,
                      tf.reduce_sum(matrix_pred_values) / tf.reduce_sum(matrix_true))

    return result


def min_max_scale(data):
    data_scaled = (data - np.min(data)) / ((np.max(data) - np.min(data)) + 1e-05)

    return data_scaled


def inference_metrics(model, data):
    """
    Prediction Efficiency Evaluation Metrics

    Args:
    - model: trained model
    - data: [protein data, smiles data, kd values]

    """

    start = time.time()
    pred_values = model.predict([data[0], data[1]])
    end = time.time()
    inf_time = end - start

    metrics = {'MSE': mse(data[2], pred_values), 'RMSE': mse(data[2], pred_values, squared=False),
               'CI': c_index(data[2], pred_values).numpy(), 'R2': r2s(data[2], pred_values),
               'Spearman': stats.spearmanr(data[2], pred_values)[0], 'Time': inf_time}

    return metrics

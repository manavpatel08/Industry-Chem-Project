"""Custom loss functions for XGBoost — handles class imbalance."""

import numpy as np


def focal_loss_objective(y_pred: np.ndarray, dtrain, gamma: float = 2.0, alpha: float = 0.25):
    """Focal loss: down-weights easy examples, focuses on hard ones."""
    y_true = dtrain.get_label()
    sigmoid = 1.0 / (1.0 + np.exp(-y_pred))
    p_t = np.where(y_true == 1, sigmoid, 1 - sigmoid)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    focal_weight = alpha_t * (1 - p_t) ** gamma

    grad = focal_weight * (sigmoid - y_true)
    hess = focal_weight * sigmoid * (1 - sigmoid)
    return grad, hess


def weighted_logloss_objective(y_pred: np.ndarray, dtrain, weight_pos: float = 2.0):
    """Weighted log-loss giving more weight to the minority (active) class."""
    y_true = dtrain.get_label()
    sigmoid = 1.0 / (1.0 + np.exp(-y_pred))
    weights = np.where(y_true == 1, weight_pos, 1.0)

    grad = weights * (sigmoid - y_true)
    hess = weights * sigmoid * (1 - sigmoid)
    return grad, hess

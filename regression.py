# -*- coding: utf-8 -*-
import sklearn.linear_model
import numpy as np
import scipy as sp
from scipy import stats


def one_shot_regression(X, y, lamb):
    """Performs a single-shot ridge regression

    Comments:
        Utilizes sklearn.linear_model.Ridge to return a weight vector for the
        regression  model y = w*biased_X + epsilon

    Args:
        X (float) : Training data of shape [n_samples, n_features]
        y (float) : Target values of shape [n_samples]
        lamb      : Regularization parameter lambda

    Returns:
        beta_vector (float) : weight vector of shape [n_featues + 1]
      """
    clf = sklearn.linear_model.Ridge(
            alpha=lamb,
            fit_intercept=True,
            normalize=False,
            copy_X=True,
            max_iter=None,
            tol=0.001,
            solver='auto',
            random_state=None)

    result = clf.fit(X, y)
    beta_vector = np.insert(result.coef_, 0, result.intercept_)

    return beta_vector


def y_estimate(biased_X, beta_vector):
    """Returns the target estimates (predicted values of the target)

    Note:
        y_pred = beta * biased_X

    Args:
        biased_X (float)    :
        beta_vector (float) :

    Returns:
        numpy array of shape [n_samples]
    """
    return np.dot(beta_vector, np.matrix.transpose(biased_X))


def sum_squared_error(biased_X, y, beta_vector):
    """Calculates the sum of squared errors

    Args:
        biased_X (float)    : array of shape [n_features + 1, n_samples]
            y               : target values of shape [n_samples]
        beta_vector (float) : array of shape [n_features + 1, 1]

    Returns:
        a scalar value
    """
    return np.sum(np.square(y - y_estimate(biased_X, beta_vector)))


def sum_squared_total(y):
    """Calculates the total sum of squares

    Args:
        y (float) : Target values of shape [n_samples]

    Returns:
        scalar value
    """
    return np.sum(np.square(y - np.mean(y)))


def r_square(biased_X, y, beta_vector):
    """Calculates R-squared value
    Args:
        biased_X (float)    :
        y (float)           :
        beta_vector (float) :

    Returns:
        scalar
    """
    SSE = sum_squared_error(biased_X, y, beta_vector)
    SST = sum_squared_total(y)

    return 1 - SSE/SST


def beta_var_covar_matrix(biased_X, y, beta_vector):
    """Calculates the variance-covariance matrix of the coefficient vector

    Comments:
        Var(beta) = (sigma^2)*(X'*X)^(-1)
        Var(beta) = MSE*(X'*X)^(-1)

    Args:
        Args:
        biased_X (float)    : Augmented X of shape [n_samples, n_features + 1]
        y (float)           : Target vector of shape [n_samples]
        beta_vector (float) : Coefficient array of shape [n_features + 1]

    Returns:
        var_covar_beta (float) : Square matrix of size [n_features + 1]
    """
    dof = len(y) - len(beta_vector)
    SSE = sum_squared_error(biased_X, y, beta_vector)
    MSE = (1/dof) * SSE

    var_covar_beta = MSE * sp.linalg.inv(np.dot(biased_X.T, biased_X))

    return var_covar_beta


def t_value(biased_X, y, beta_vector):
    """
    Args:
        biased_X (float)    :
        y (float)           :
        beta_vector (float) :

    Returns:
        ts_beta (float) : t-values for each beta of shape [n_features + 1]
    """
    beta_variance = beta_var_covar_matrix(biased_X, y, beta_vector)

    se_beta = np.sqrt(beta_variance.diagonal())
    ts_beta = beta_vector/se_beta

    return ts_beta


def t_to_p(dof, ts_beta):
    """
    Args:
        dof (int)       : len(y) - len(beta_vector)
        ts_beta (float) : array of shape [n_features +  1]

    Returns:
        p_values (float) of shape [n_features + 1]

    Comments:
        t to p value transformation(two tail)
    """
    p_values = [2*stats.t.sf(np.abs(t), dof) for t in ts_beta]

    return p_values

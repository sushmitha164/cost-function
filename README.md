# cost-function

import numpy as np

def compute_cost(X, y, theta):
    """
    Compute cost for linear regression.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1), where m is the number of examples,
        and n is the number of features. We assume a vector of one's already
        appended to the features so we have n+1 columns.
    y : array_like
        The values of the function at each data point. This is a vector of
        shape (m, ).
    theta : array_like
        The parameters for the regression function. This is a vector of
        shape (n+1, ).

    Returns
    -------
    J : float
        The value of the regression cost function.
    """
    # Number of training examples
    m = len(y)

    # Hypothesis function
    h = np.dot(X, theta)

    # Cost function
    J = np.sum((h - y) ** 2) / (2 * m)

    return J

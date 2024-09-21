import numpy as np

def matrix_ols(X, y):
    """
    Solves OLS using matrix inversion

    Parameters:
    X (numpy array): n x (p + 1) matrix of independent observations. P is the number of independent variables and each column represents n observations for the variable.
    y (numpy vector): n x 1 column vector of dependent observations
    """

    covar_matrix = X.T @ X

    if np.linalg.det(covar_matrix) == 0:
        raise ValueError("Covariance Matrix is Singular")

    xtx_inverse = np.linalg.inv(covar_matrix)

    beta_hat = xtx_inverse @ X.T @ y

    return beta_hat

def gram_schimdt_ols(X, y):
     """
    Solves OLS using succesive orthogonalization

    Parameters:
    X (numpy array): n x (p + 1) matrix of independent observations. p is the number of independent variables and each column represents n observations for the variable.
    y (numpy vector): n x 1 column vector of dependent observations
    """

    Q, R = np.linalg.qr(X)

    beta_hat = np.linalg.inv(R) @  Q.T @ y

def single_var_no_intercept_regression(x, y):

    beta_hat = (x.T @ y) / (x.T @ x)

    return beta_hat

def single_var_intercept_regression(x, y):
    z = x - np.mean(x)

    beta_hat = (z.T @ y) / (z.T @ z)

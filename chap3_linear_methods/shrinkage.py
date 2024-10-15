import numpy as np

def ridge (X, y, lamb = 0):
    """
    Solves Ridge using matrix inversion
    """
    intercept_column = np.ones(X.shape[0], 1)
    X  = np.hstack((intercept_column, X))

    covar_matrix = X.T @ X

    xtx_inverse = np.linalg.inv(covar_matrix + lamb * np.identity(covar_matrix.size[0]))
    
    beta_hat = xtx_inverse @ X.T @ y

    return beta_hat




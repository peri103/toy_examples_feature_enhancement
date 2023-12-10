from base_svm import BaseSVM, np, libsvm, _svm_types, _kernel_types

class SVR(BaseSVM):
    """
    Support Vector Regression.
    Parameters
    ----------
    X : array-like, shape = [N, D]
        Training vector
    Y : array, shape = [N]
        Target vector relative to X
    
    """
    def __init__(self, svm='epsilon_svr', kernel='rbf', degree=3,
                 gamma=0.0, coef0=0.0, cache_size=100.0, eps=1e-3,
                 C=1.0, nr_weight=0, nu=0.5, p=0.1, shrinking=1,
                 probability=0):
        BaseSVM.__init__(self, svm, kernel, degree, gamma, coef0,
                         cache_size, eps, C, nr_weight, nu, p,
                         shrinking, probability)
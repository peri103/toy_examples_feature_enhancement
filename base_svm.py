import numpy as np
import libsvm
_kernel_types = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
_svm_types = ['c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr']
class BaseSVM(object):
    """
    Base class for classifiers that use support vector machine.
    Should not be used directly, use derived classes instead
    Parameters
    ----------
    X : array-like, shape = [N, D]
        It will be converted to a floating-point array.
    y : array, shape = [N]
        target vector relative to X
        It will be converted to a floating-point array.
    """

    def __init__(self, svm, kernel, degree, gamma, coef0, cache_size,
                 eps, C, nr_weight, nu, p, shrinking, probability):
        self.svm = _svm_types.index(svm)
        self.kernel = _kernel_types.index(kernel)
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.cache_size = cache_size
        self.eps = eps
        self.C = C
        self.nr_weight = 0
        self.nu = nu
        self.p = p
        self.shrinking = shrinking
        self.probability = probability

    def fit(self, X, y):
        """
        should empty arrays created be order='C' ?
        """
        X = np.asanyarray(X, dtype=np.float, order='C')
        y = np.asanyarray(y, dtype=np.float, order='C')

        # check dimensions
        if X.shape[0] != y.shape[0]: raise ValueError("Incompatible shapes")

        if (self.gamma == 0): self.gamma = 1.0/X.shape[0]
        self.coef_, self.rho_, self.support_, self.nclass_, self.nSV_, self.label_  = \
             libsvm.train_wrap(X, y, self.svm, self.kernel, self.degree,
                 self.gamma, self.coef0, self.eps, self.C, self.nr_weight,
                 np.empty(0, dtype=np.int), np.empty(0, dtype=np.float), self.nu,
                 self.cache_size, self.p, self.shrinking, self.probability)
        return self

    def predict(self, T):
        T = np.asanyarray(T, dtype=np.float, order='C')
        return libsvm.predict_from_model_wrap(T, self.support_,
                      self.coef_, self.rho_, self.svm,
                      self.kernel, self.degree, self.gamma,
                      self.coef0, self.eps, self.C, self.nr_weight,
                      np.empty(0, dtype=np.int), np.empty(0,
                      dtype=np.float), self.nu, self.cache_size,
                      self.p, self.shrinking, self.probability,
                      self.nclass_, self.nSV_, self.label_)
def predict(X, y, T, svm='c_svc', kernel='rbf', degree=3,
                 gamma=0.0, coef0=0.0, cache_size=100.0, eps=1e-3,
                 C=1.0, nr_weight=0, nu=0.5, p=0.1, shrinking=1,
                 probability=0):
    """
    Shortcut that does fit and predict in a single step.
    Should be faster than instatating the object, since less copying is done.
    Parameters
    ----------
    X : array-like
        data points
    y : array
        targets
    T : array
        test points
    Optional Parameters
    -------------------
    TODO
    Examples
    --------
    """
    X = np.asanyarray(X, dtype=np.float, order='C')
    y = np.asanyarray(y, dtype=np.float, order='C')
    T = np.asanyarray(T, dtype=np.float, order='C')
    if X.shape[0] != y.shape[0]: raise ValueError("Incompatible shapes")
    return libsvm.predict_wrap(X, y, T, _svm_types.index(svm),
                               _kernel_types.index(kernel),
                               degree, gamma, coef0, eps, C,
                               nr_weight, np.empty(0, dtype=np.int),
                               np.empty(0, dtype=np.float), nu,
                               cache_size, p, shrinking, probability)
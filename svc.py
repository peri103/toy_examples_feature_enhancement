from base_svm import BaseSVM, np, libsvm, _svm_types, _kernel_types

class SVC(BaseSVM):
    """
    Support Vector Classification
    Implementats C-SVC, nu-SVC
    Parameters
    ----------
    X : array-like, shape = [nsamples, nfeatures]
        Training vector, where nsamples in the number of samples and
        nfeatures is the number of features.
    Y : array, shape = [nsamples]
        Target vector relative to X
    impl : string, optional
        SVM implementation to choose from. This refers to different
        formulations of the SVM optimization problem.
        Can be one of 'c_svc', 'nu_svc'. By default 'c_svc' will be chosen.
    nu: float, optional
        An upper bound on the fraction of training errors and a lower
        bound of the fraction of support vectors. Should be in the
        interval (0, 1].
        By default 0.5 will be taken.
        Only available is impl is set to 'nu_svc'
    kernel : string, optional
         Specifies the kernel type to be used in the algorithm.
         one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'.
         If none is given 'rbf' will be used.
    degree : int, optional
        degree of kernel function
        is significant only in POLY, RBF, SIGMOID
    Members
    -------
    support_ : array-like, shape = [nSV, D]
        estimated support vectors.
        where nSV is the number of support vectors, D is the dimension
        of the underlying space.
    coef_ : array
        coefficient of the support vector in the decission function.
    rho_ : array
        constants in decision function
    Examples
    --------
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> clf = SVM()
    >>> clf.fit(X, y)    #doctest: +ELLIPSIS
    <scikits.learn.svm.svm.SVM object at 0x...>
    >>> print clf.predict([[-0.8, -1]])
    [ 1.]
    See also
    --------
    http://scikit-learn.sourceforge.net/doc/modules/svm.html
    http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf
    """
    def __init__(self, impl='c_svc', kernel='rbf', degree=3,
                 gamma=0.0, coef0=0.0, cache_size=100.0, eps=1e-3,
                 C=1.0, nr_weight=0, nu=0.5, p=0.1, shrinking=1,
                 probability=0):

        BaseSVM.__init__(self, impl, kernel, degree, gamma, coef0,
                         cache_size, eps, C, nr_weight, nu, p,
                         shrinking, probability)
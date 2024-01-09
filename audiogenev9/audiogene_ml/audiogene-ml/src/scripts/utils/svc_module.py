from sklearn.svm import SVC


class SVC_Linear(SVC):
    def __init__(self, kernel = 'linear', C=1, verbose=3, random_state = 3,
                 probability = False, max_iter = 1000, tol = 1e-2):
        SVC.__init__(self, C = C, kernel = kernel, verbose = verbose, random_state = random_state,
                     probability = probability, max_iter=max_iter, tol=tol)

class SVC_Poly(SVC):
    def __init__(self, kernel = 'poly', C = 1, degree = 3, coef0 = 0, gamma = 1,
                 probability = False, verbose=3, random_state = 3, max_iter = 1000, tol = 1e-2):
        SVC.__init__(self, C = C, kernel = kernel, degree = degree, gamma = gamma,
                     coef0 = coef0 , probability = probability,
                     verbose = verbose, random_state = random_state, max_iter=max_iter, tol=tol)

class SVC_rbf(SVC):
    def __init__(self, kernel = 'rbf', C = 1, gamma = 1, probability = False,
                 random_state = None, verbose = 3, max_iter = 1000, tol = 1e-2):
        SVC.__init__(self, C = C, kernel = kernel, gamma = gamma,
                     probability = probability, verbose = verbose,
                     random_state = random_state, max_iter=max_iter, tol=tol)
from sklearn import svm
import numpy as np
from sklearn.svm import LinearSVC

def svm_classify(x, y, C=0.1, max_iter=1000, tol=1e-4):
    '''
    FUNC: train SVM classifier with input data x and label y
    ARG:
        - x: input data, HOG features
        - y: label of x, face or non-face
    RET:
        - clf: a SVM classifier using sklearn.svm. (You can use your favorite
               SVM library but there will be some places to be modified in
               later-on prediction code)
    '''
    #########################################
    ##          you code here              ##
    #########################################
    
    print("START: svm_classify")
    
    clf = LinearSVC(C=C, class_weight=None, dual=True, fit_intercept=True,
                    intercept_scaling=1, loss='squared_hinge', max_iter= max_iter,
                    multi_class='ovr', penalty='l2', random_state=0, tol= tol,
                    verbose=0)
    #clf = calibration.CalibratedClassifierCV(clf, method='sigmoid')
    clf.fit(x, y)
    
    print("DONE: svn_classify")

    #########################################
    ##          you code here              ##
    #########################################

    return clf
import numpy as np

def cross_product(X1, X2):
    """ Cross product of features in matrices X1 and X2
    """
    n = np.shape(X1)[0]
    assert n == np.shape(X2)[0]
    return (X1.reshape(n,1,-1) * X2.reshape(n,-1,1)).reshape(n,-1) 


def filesafe(str):
    return "".join([c for c in str if c.isalpha() or c.isdigit() or c==' ']).rstrip().replace(' ', '_')

import pandas as pd
import numpy as np
import time as time
from __future__ import division

def jaccard_dist(a,b):
    '''
    Computes Jaccard distance between two vectors a and b

    INPUT:
    a: (numpy array)
    b: (numpy array)

    OUTPUT:
    dist: (float) Jaccard distance
    '''

    intersect = np.dot(a,b)
    s_ab = a + b
    union = len(s_ab[s_ab > 0])

    if union:
        sim = intersect/union
    else:
        sim = 0

    dist = 1 - sim

    return dist

def jaccard_mat(df, axis=1, timer=False):
    '''
    Computes Jaccard distance between vectors in a matrix

    INPUT:
    df: (pandas DataFrame) Utiility matrix of (usrs, items)
    axis: (int) Indicates similarity across rows (0) or columns (1)

    OUTPUT:
    sim: (pandas DataFrame) Similarity matrix
    '''
    start_time = time()

    df_copy = df.copy()
    users = df_copy.pop('UserID')
    items = np.array(df_copy.columns)

    dimen = df_copy.shape[axis]

    zero_data = np.zeros((dimen,dimen))

    if axis:
        sim = pd.DataFrame(zero_data, columns=items, dtype=float)
    else:
        sim = pd.DataFrame(zero_data, columns=users, dtype=float)
        df_copy = df_copy.T

    for index0 in xrange(dimen):
        vec0 = df_copy.iloc[:,index0]
        for index1 in xrange(dimen):
            vec1 = df_copy.iloc[:,index1]
            sim.iloc[index0,index1] = jaccard_dist(vec0,vec1)

    if timer:
        print "Jaccard distance computation took: ", time() - start_time

    return sim


def cos_dist(a,b):
    '''
    Conputes Cosine distance between two vectors a and b

    INPUT:
    a: (numpy array)
    b: (numpy array)

    OUTPUT:
    dist: (float) cosine distance
    '''

    a_norm = np.sqrt(np.dot(a,a))
    b_norm = np.sqrt(np.dot(b,b))

    dist = np.dot(a,b)/(a_norm * b_norm)

    return dist

    # A = set(a)
    # B = set(b)
    #
    # sim = len(A & B)/len(A | B)
    #
    # dist = 1 - sim

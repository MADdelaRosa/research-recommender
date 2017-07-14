from __future__ import division
import pandas as pd
import numpy as np
from time import time

def jaccard_dist(a,b):
    '''
    Computes Jaccard distance between two vectors a and b

    INPUT:
    a: (numpy array)
    b: (numpy array)

    OUTPUT:
    dist: (float) Jaccard distance
    '''

    if ((a == 0).all() and (b == 0).all()):
        sim = 1.0
    else:
        intersect = np.dot(a,b)
        s_ab = a + b
        union = float(len(s_ab[s_ab > 0]))

        if union != 0:
            sim = intersect/union
        else:
            sim = 0.0

    dist = 1.0 - sim

    return dist

def jaccard_mat(df, axis=1, timer=False):
    '''
    Computes Jaccard distance between vectors in a matrix

    INPUT:
    df: (pandas DataFrame) Utiility matrix of (usrs, items)
    axis: (int) Indicates similarity across rows (0) or columns (1)
    timer: (bool) Whether to measure runtime

    OUTPUT:
    sim: (pandas DataFrame) Similarity matrix
    '''
    start_time = time()

    df_copy = df.copy()

    if df.columns[0] == 'UserID':
        users = df_copy.pop('UserID')
    else:
        users = df_copy.index.values

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
        for index1 in xrange(index0 + 1, dimen):
            vec1 = df_copy.iloc[:,index1]
            jd = jaccard_dist(vec0,vec1)
            sim.iloc[index0,index1] = jd
            sim.iloc[index1,index0] = jd

    if timer:
        print "Jaccard distance computation took: ", time() - start_time

    return sim

def jaccard_update(utility, sim, utility_entry, axis=1, return_util=True, timer=False):
    '''
    Updates similarity matrix after removing a given data point from utility matrix

    INPUT:
    utility: (pandas DataFrame) Original utliity matrix
    sim: (pandas DataFrame) Original similarity matrix
    utility_entry: (tuple) (usr,content) entry to remove in utility matrix (int, int)
    axis: (int) Indicates similarity across rows (0) or columns (1)
    timer: (bool) Whether to measure runtime

    OUTPUT:
    sim: (pandas DataFrame) New similarity matrix
    '''

    start_time = time()

    util_copy = utility.copy()
    sim_copy = sim.copy()
    user = utility_entry[0]
    # item = str(utility_entry[1])
    item = utility_entry[1]
    util_copy.loc[util_copy.UserID == user, [item]] = 0
    users = util_copy.pop('UserID')
    # util_copy = util_copy.drop(['UserID'], axis=1)

    dimen = util_copy.shape[axis]

    if axis == 0:
        util_copy = util_copy.T

    index = util_copy.columns.get_loc(item)
    vec1 = util_copy.iloc[:,index]
    for i in xrange(dimen):
        if i != index:
            vec2 = util_copy.iloc[:,i]
            jd = jaccard_dist(vec1,vec2)
            sim_copy.iloc[index,i] = jd
            sim_copy.iloc[i,index] = jd

    if return_util:
        utility.loc[utility.UserID == user, [item]] = 1
        util_copy.insert(0,'UserID',users)
    # del util_copy

    if timer:
        print "Jaccard update took: ", time() - start_time

    if return_util:
        return sim_copy, sim, util_copy, utility
    else:
        return sim_copy

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

def pairwise_jaccard(X):
    """Computes the Jaccard distance between the rows of `X`.
    """
    # X = X.astype(bool).astype(int)

    intrsct = X.dot(X.T)
    row_sums = intrsct.diagonal()
    unions = row_sums[:,None] + row_sums - intrsct
    dist = 1.0 - intrsct / unions
    return dist

# st = time()
# pairwise_jaccard(matts.T)
# print "Time: ", time() - st

# A = set(a)
# B = set(b)
#
# sim = len(A & B)/len(A | B)
#
# dist = 1 - sim

# st = time()
# np.fill_diagonal(mat1.values, 0)
# print "Took: ", time() - st

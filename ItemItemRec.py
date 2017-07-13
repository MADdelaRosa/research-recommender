from __future__ import division
import pandas as pd
import numpy as np
from time import time
from collaborative_filtering import jaccard_dist, cos_dist, jaccard_mat, jaccard_update

class ItemItemRec(object):

    def __init__(self, n_size):

        self.neighborhod = neighborhood

    def fit(self, utility_matrix):

        self.utility_matrix = utility_matrix
        self.n_users = utility_matrix.shape[0]
        self.n_items = utility_matrix.shape[1]
        self.sim_mat = jaccard_mat(self.utility_matrix,axis=1)
        self._set_neighborhoods()

    def _set_neighborhoods(self):

        sim_indeces = np.argsort(self.sim_mat, 1).values
        self.neighborhoods = sim_indeces[:, -self.neighborhod:]

    def pred_one_user(self, user, timer=False):

        start_time = time()
        items_rated = self.utility_matrix[utility_matrix.UserID == user]. \
            drop(['UserID'].axis=1).values.nonzero()[1]

        # Just initializing so we have somewhere to put rating preds
        out = np.zeros(self.n_items)
        for item_to_rate in xrange(self.n_items):
            relevant_items = np.intersect1d(self.neighborhoods[item_to_rate],
                                            items_rated,
                                            assume_unique=True)  # assume_unique speeds up intersection op
            out[item_to_rate] = self.utility_matrix[user, relevant_items] * \
                self.sim_mat[item_to_rate, relevant_items] / \
                self.sim_mat[item_to_rate, relevant_items].sum()

        if timer:
            print "Prediction time: {} seconds".format(time()-start_time)

        cleaned_out = np.nan_to_num(out)
        return cleaned_out

from __future__ import division
import pandas as pd
import numpy as np
from time import time
from collaborative_filtering import jaccard_dist, cos_sim, jaccard_mat, jaccard_update
from baseline import recommend_random_fav

class ItemItemRec(object):

    def __init__(self, n_size):

        self.n_size = n_size

    def fit(self, utility_matrix, prev_fit=True, fav=True):

        self.utility_matrix = utility_matrix
        self.n_users = utility_matrix.shape[0]
        self.n_items = utility_matrix.shape[1] - 1
        self.col_labels = utility_matrix.columns.values
        self.usr_ids = utility_matrix.UserID.values
        self._load_favorite_dist()
        if prev_fit:
            self._load_prev_similarity_matrix(fav)
        else:
            self.sim_mat = jaccard_mat(self.utility_matrix,axis=1)
        self._set_neighborhoods()

    def _load_prev_similarity_matrix(self,fav):
        if fav:
            self.sim_mat = pd.read_csv('data/modified_data/item_item_fav.csv')
        else:
            self.sim_mat = pd.read_csv('data/modified_data/item_item_dld.csv')

    def _load_favorite_dist(self):
        self.fav_dist = pd.read_csv('data/modified_data/user-favorites.csv').content_id

    def _set_neighborhoods(self):

        sim_indeces = np.argsort(self.sim_mat, 1).values
        # self.neighborhoods = sim_indeces[:, -self.n_size:]
        self.neighborhoods = sim_indeces[:, 0:self.n_size]

    def rec_one_user(self, user, rec_num = 3, wide = True, timer=False):

        start_time = time()
        user_ind = self.utility_matrix[self.utility_matrix.UserID == user].index[0]
        user_items = self.utility_matrix[self.utility_matrix.UserID == user]. \
            drop(['UserID'],axis=1).values.nonzero()[1]

        item_score = np.zeros(self.n_items)
        for item in xrange(self.n_items):
            item_intersection = np.intersect1d(self.neighborhoods[item],
                user_items, assume_unique=True)

            item_score[item] = len(item_intersection)

        if item_score.any():
            top_indeces = np.argsort(item_score)

            recommend = []
            i = -1
            while (len(recommend) < rec_num) and (abs(i) <= len(top_indeces)):
                index = top_indeces[i] + 1 # account for UserID column

                if self.utility_matrix.iloc[user_ind, index] == 0:
                    recommend.append(index)

                if wide:
                    i -= 1
                else:
                    hood = self.neighborhoods[index-1]
                    # for j in xrange(len(hood)-1,-1,-1):
                    for j in xrange(len(hood)):
                        if len(recommend) < rec_num:
                            hood_ind = hood[j] + 1 # account for UserID column
                            if self.utility_matrix.iloc[user_ind, hood_ind] == 0:
                                recommend.append(hood_ind)
                    i -= 1

            recommend = np.array(recommend)
            # print recommend

            rec_items = self.col_labels[recommend]
        else:
            rec_items = recommend_random_fav(self.fav_dist,rec_num)

        if timer:
            print "Recommender calculation time: {} seconds".format(time()-start_time)

        return rec_items

    def rec_all_users(self, rec_num = 3, wide = True, timer=False):

        start_time = time()

        all_recs = [self.rec_one_user(user_id, rec_num, wide, timer)
            for user_id in self.usr_ids]

        if timer:
            print "Recommender calculation time: {} seconds".format(time()-start_time)

        return np.array(all_recs)

    def leave_one_out(self, user, item, rec_num = 3, wide = True, timer=False):

        start_time = time()

        entry = (user, item)

        self.sim_mat, prev_sim, self.utility_matrix, prev_util = \
            jaccard_update(self.utility_matrix, self.sim_mat, entry, axis=1)


        print "Util val: "
        print self.utility_matrix.loc[self.utility_matrix.UserID == user, [item]]
        recommend = self.rec_one_user(user, rec_num, wide)
        print "For item: ", item
        print "Recommended docs: ", recommend

        rec_entry = np.in1d(item, recommend)
        print "Recommended item? ", rec_entry
        self.sim_mat = prev_sim
        self.utility_matrix = prev_util
        del prev_sim
        del prev_util
        print "Util val: "
        print self.utility_matrix.loc[self.utility_matrix.UserID == user, [item]]

        if timer:
            print "Prediction time: {} seconds".format(time()-start_time)

        return rec_entry[0]

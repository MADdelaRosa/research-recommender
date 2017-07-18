from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ItemItemRec import ItemItemRec
from collaborative_filtering import jaccard_dist, jaccard_mat, jaccard_update, cos_sim
from baseline import recommend_random_fav
from profiling import update_user_profile, content_profile_recommender
from time import time
import gc

TEST_0 = False
VALIDATE = True

user_profiles = pd.read_csv('data/modified_data/user_profile')
item_features = pd.read_csv('data/modified_data/item_features.csv')


if TEST_0:

    '''
    Test ItemItemRec with larger dataset:
    '''

    utility = pd.read_csv('data/modified_data/utility_dld_matrix.csv')
    mask_dlds = (utility.drop('UserID',1) > 0).any(axis=1)
    utility_dld = utility[mask_dlds].reset_index(drop=True)
    del utility

    # similarity = pd.read_csv('data/modified_data/item_item_dld.csv')

    recommender = ItemItemRec(n_size=50)
    recommender.fit(utility_dld, prev_fit=True, fav=False)

    users = utility_dld.UserID
    user = np.random.choice(users)

    recs_w = recommender.rec_one_user(user,timer=True)
    print "Recommended items for UserID {} are: {}".format(user,recs_w)

    user_items_ind = utility_dld[utility_dld.UserID == user].drop(['UserID'],axis=1).values.nonzero()[1]
    user_items = recommender.col_labels[user_items_ind]
    one_out = np.random.choice(user_items)
    test = recommender.leave_one_out(user,one_out,rec_num=5,timer=True)
    print "For UserID = {}, does the recommender recommend item {}? {}".format(user,one_out,test)

if VALIDATE:

    start_time = time()

    '''
    Validate ItemItemRec:
    '''

    utility = pd.read_csv('data/modified_data/utility_dld_matrix.csv')
    mask_dlds = (utility.drop('UserID',1) > 0).any(axis=1)
    utility_dld = utility[mask_dlds].reset_index(drop=True)
    util_mat_copy = utility_dld.copy()
    del utility

    # Choose top users that account for a predetermined percentage of signal:

    user_dlds = pd.read_csv('data/modified_data/top_downloads_users.csv')
    # cutoff = 500    # corresponds to 30% of download signal
    # cutoff = 250
    # cutoff = 10
    cutoff = 10
    top_users = user_dlds.UserID.values[0:cutoff]
    # top_users = user_dlds.UserID.values[cutoff:2*cutoff]

    # Initialize recommender:

    # recommender = ItemItemRec(n_size=290)
    recommender = ItemItemRec(n_size=100)
    recommender.fit(utility_dld, prev_fit=True, fav=False)

    prediction = []
    # print "Start looping over all top users"
    count = 0
    for user in top_users:
        count += 1
        print count

        print "This is user ", user

        # print "Get user item index"
        user_items_ind = utility_dld[utility_dld.UserID == user].drop(['UserID'],
            axis=1).values.nonzero()[1]

        print "Pick out random item"
        one_out = np.random.choice(recommender.col_labels[user_items_ind])
        print one_out

        # print "Launch recommender"
        test_0, rec_set = recommender.leave_one_out(user,one_out,rec_num=100,\
            wide=True,timer=False)

        # garbage collect
        if count % 5 == 0:
            collected = gc.collect()
            print "Garbage collector: collected %d objects." % (collected)

        # Content based recommendation:

        update_user_profile(user_profiles, util_mat_copy, \
            item_features, (user,one_out), out=True, timer=False)

        new_set = content_profile_recommender(user,rec_set,user_profiles,\
            item_features,rec_num = 5,timer=False)

        print "Content recommender narrows it down to:"
        print new_set

        update_user_profile(user_profiles, util_mat_copy, \
            item_features, (user,one_out), out=False, timer=False)

        test = np.in1d(one_out, new_set)
        print "Is target in set?: ", test

        print "Append to prediction list"
        prediction.append(test)
        # prediction.append(test_0)

    score = (1*np.array(prediction)).sum()/len(prediction)

    print "Recommenders predict downloads with a score of: ", score

    print "Calculation time: {} seconds".format(time()-start_time)

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ItemItemRec import ItemItemRec
from collaborative_filtering import jaccard_dist, jaccard_mat, jaccard_update
from baseline import recommend_random_fav
from time import time
import gc

TEST_0 = False

TEST_1 = False

VALIDATE = True

if TEST_0:

    '''
    Test ItemItemRec:
    '''

    # data = np.random.choice([0,1], size=(10,10))
    # ids =  np.random.choice(range(500),size=10)
    # dat = pd.DataFrame(data, columns=range(10))
    data = np.random.choice([0,1], size=(1000,100), p=[7./8,1./8])
    ids =  np.random.choice(range(5000),size=1000)
    dat = pd.DataFrame(data, columns=range(100))
    dat.insert(0,'UserID',ids)
    print dat

    recommender = ItemItemRec(n_size=5)
    recommender.fit(dat,prev_fit=False,fav=True)

    user = np.random.choice(ids)

    recs_w = recommender.rec_one_user(user,timer=True)
    print "Recommendation, wide: ", recs_w

    recs_d = recommender.rec_one_user(user,wide=False,timer=True)
    print "Recommendation, deep: ", recs_d

    user_items = dat[dat.UserID == user].drop(['UserID'],axis=1).values.nonzero()[1]
    one_out = np.random.choice(user_items)

    test = recommender.leave_one_out(user,one_out,timer=True)
    print "Recommender recommends item {}? {}".format(one_out,test)

if TEST_1:

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
    del utility

    # Choose top users that account for a predetermined percentage of signal:

    user_dlds = pd.read_csv('data/modified_data/top_downloads_users.csv')
    # cutoff = 500    # corresponds to 30% of download signal
    cutoff = 1500
    # cutoff = 10
    # cutoff = 10
    # top_users = user_dlds.UserID.values[0:cutoff]
    top_users = user_dlds.UserID.values[cutoff:2*cutoff]

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
        test = recommender.leave_one_out(user,one_out,rec_num=50,wide=True,timer=False)

        # garbage collect
        if count % 5 == 0:
            collected = gc.collect()
            print "Garbage collector: collected %d objects." % (collected)

        # print "Append to prediction list"
        prediction.append(test)

    score = (1*np.array(prediction)).sum()/len(prediction)

    print "Recommender predicts downloads with a score of: ", score

    print "Calculation time: {} seconds".format(time()-start_time)

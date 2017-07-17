import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import gc
from sklearn.metrics.pairwise import cosine_similarity
from collaborative_filtering import cos_sim

def populate_item_profile(df, item_data, columns):
    '''
    Populates item profile matrix.

    INPUT:
    df: (pandas DataFrame) Empty item profile matrix
    item_data: (pandas DataFrame) Item metadata
    columns: (array-like) List of columns to look up category values in item_data

    OUTPUT:
    df: (pandas DataFrame) Populated item profile matrix
    '''

    content_ids = df.content_id.values

    items_in_data = item_data.content_id

    for item in content_ids:
        if (items_in_data == item).any():
            for column in columns:
                col_cat = item_data.loc[item_data.content_id == item, column].values[0]
                # print item, column, col_cat
                if col_cat != 0:
                    df.loc[df.content_id == item, col_cat] = 1

    return df

def populate_base_user_profile(df, subject_data, user_subjects):
    '''
    Populate base user profile matrix, solely from expressed preferences.

    INPUT:
    df: (pandas DataFrame) Empty user profile matrix
    subject_data: (pandas DataFrame) Subject area metadata
    user_subjects: (pandas DataFrame) Chosen user subject areas
    users: (numpy array) List of users used in analysis

    OUTPUT:
    df: (pandas DataFrame) Populated user profile matrix
    '''

    for index in xrange(len(user_subjects)):
        us_id = user_subjects.iloc[index,:][0]

        sa_id = user_subjects.iloc[index,:][1]
        subject_info = subject_data[subject_data.SubjectAreaID == sa_id].values.flatten()
        # print index, subject_info
        sa = subject_info[1]
        pr = subject_info[2]

        df.loc[df.UserID == us_id, sa] = 1
        df.loc[df.UserID == us_id, pr] = 1

    return df

def populate_user_profile(df, utility_matrix, item_profile, timer=True):
    '''
    Populates user profile matrix from usage found in utility matrix.

    INPUT:
    df: (pandas DataFrame) Empty user profile matrix
    utility_matrix: (pandas DataFrame) Utility matrix
    item_profile: (pandas DataFrame) Item profile matrix

    OUTPUT:
    df: (pandas DataFrame) Populated user profile matrix
    '''

    start_time = time()

    user_ids = df.UserID
    df.drop(['UserID'],axis=1,inplace=True)

    utility_matrix.drop(['UserID'],axis=1,inplace=True)
    content_ids = utility_matrix.columns
    u_matrix = utility_matrix.values

    item_profile.drop(['content_id'],axis=1,inplace=True)
    item_prof = item_profile.values

    content_inds = np.array(range(u_matrix.shape[1]))

    dlds_user = u_matrix > 0

    for index in xrange(len(u_matrix)):

        dlds = content_inds[dlds_user[index]]
        number_dlds = len(dlds)

        feat_vecs = item_prof[dlds]

        scores = np.sum(feat_vecs,axis=0)/float(number_dlds)

        df.iloc[index,:] = df.iloc[index,:] + scores

    df.insert(0, 'UserID', user_ids)
    utility_matrix.insert(0, 'UserID', user_ids)
    item_profile.insert(0, 'content_id', content_ids)

    if timer:
        print "User profile matrix time: ", (time() - start_time)

    return df

# def update_user_profile_old(user_profile, utility_matrix, item_profile, entry, out=True, timer=True):
#     '''
#     Updates entries in user profile matrix when changing single value of
#     user/item interaction (donwnload).
#
#     INPUT:
#     user_profile: (pandas Dataframe) User profile matrix
#     utility: (pandas Dataframe) Utility matrix
#     item_profile: (pandas DataFrame) Item profile matrix
#     entry: (tuple) (user,content) entry to remove in utility matrix (int, int)
#     out: (bool) Whether the change is taking out (True) or putting back in (False)
#
#
#     OUTPUT:
#     user_profile: (pandas Dataframe) Updated profile matrix
#     '''
#     start_time = time()
#
#     user = entry[0]
#     item = entry[1]
#
#     user_ids = df.UserID
#     user_index = user_profile[user_profile == user].index
#     user_profile.drop(['UserID'],axis=1,inplace=True)
#
#     utility_matrix.drop(['UserID'],axis=1,inplace=True)
#     content_ids = utility_matrix.columns
#
#     item_profile.drop(['content_id'],axis=1,inplace=True)
#
#     content_inds = np.array(range(len(content_ids)))
#     user_util = utility_matrix[utility_matrix == user].values
#
#     dlds_user = user_util > 0
#     dlds = content_inds[dlds_user]
#     number_dlds_old = len(dlds)
#     number_dlds_new = number_dlds_old - 1
#
#     item_vec = item_profile[item_profile == item].values
#     relevant_features = item_vec > 0
#     changed_features = content_inds[relevant_features]
#
#     for feature in changed_features:
#         old_score = user_profile.iloc[user_index,feature]
#
#         new_score = ((old_score*number_dlds_old) - 1)/number_dlds_new
#
#         user_profile.iloc[user_index, feature] = new_score
#
#
#     user_profile.insert(0, 'UserID', user_ids)
#     utility_matrix.insert(0, 'UserID', user_ids)
#     item_profile.insert(0, 'content_id', content_ids)
#
#     if timer:
#         print "User profile matrix time: ", (time() - start_time)
#
#     return user_profile

def update_user_profile(user_profile, utility_matrix, item_profile, entry, out=True, timer=True):
    '''
    Updates entries in user profile matrix when changing single value of
    user/item interaction (donwnload).

    INPUT:
    user_profile: (pandas Dataframe) User profile matrix
    utility: (pandas Dataframe) Utility matrix
    item_profile: (pandas DataFrame) Item profile matrix
    entry: (tuple) (user,content) entry to remove in utility matrix (int, int)
    out: (bool) Whether the change is taking out (True) or putting back in (False)


    OUTPUT:
    user_profile: (pandas Dataframe) Updated profile matrix
    '''
    start_time = time()

    user = entry[0]
    item = entry[1]

    if out:
        sub_value = 0
    else:
        sub_value = 1

    user_ids = user_profile.UserID
    user_index = user_profile[user_profile.UserID == user].index[0]
    user_profile.drop(['UserID'],axis=1,inplace=True)

    content_ids = item_profile.content_id
    item_index = item_profile[item_profile.content_id == item].index[0]
    item_profile.drop(['content_id'],axis=1,inplace=True)
    item_prof = item_profile.values

    # Update utility matrix:
    utility_matrix.loc[utility_matrix.UserID == user, str(item)] = sub_value
    utility_matrix.drop(['UserID'],axis=1,inplace=True)
    u_matrix = utility_matrix.values
    user_util = u_matrix[user_index]

    content_inds = np.array(range(len(content_ids)))

    dlds_user = user_util > 0

    dlds = content_inds[dlds_user]
    number_dlds = len(dlds)

    feat_vecs = item_prof[dlds]

    scores = np.sum(feat_vecs,axis=0)/float(number_dlds)

    user_profile.iloc[user_index,:] = scores

    user_profile.insert(0, 'UserID', user_ids)
    utility_matrix.insert(0, 'UserID', user_ids)
    item_profile.insert(0, 'content_id', content_ids)

    if timer:
        print "User profile matrix time: ", (time() - start_time)

def content_profile_recommender(user,items,user_profile,item_profile,rec_num = 3,timer=False):
    '''
    Recommends items based on content similarity between user and item
    profiles.

    INPUT:
    user: (int) User ID of user to recommend for
    items: (array-like) Content ID of items of interest
    user_profile: (pandas Dataframe) User profile matrix
    item_profile: (pandas DataFrame) Item profile matrix
    rec_num: (int) Number of recommendations to return


    OUTPUT:
    recommend: (numpu array) List of recommended items
    '''

    start_time = time()

    items_given = len(items)
    similarities = np.zeros(items_given)

    user_ids = user_profile.UserID
    user_index = user_profile[user_profile.UserID == user].index[0]
    user_profile.drop(['UserID'],axis=1,inplace=True)
    user_vector = user_profile.iloc[user_index,:].values

    content_ids = item_profile.content_id
    # item_index = item_profile[item_profile.content_id == item].index[0]
    item_profile.drop(['content_id'],axis=1,inplace=True)
    item_prof = item_profile.values

    for i, item in enumerate(items):
        item_index = np.where(content_ids == str(item))[0][0]

        item_vect = item_prof[item_index]

        similarities[i] = cos_sim(user_vector,item_vect)

    # Rank similarities:

    indeces = np.argsort(similarities)

    ranked_items = items[indeces]

    recommend = ranked_items[-rec_num:]

    user_profile.insert(0, 'UserID', user_ids)
    item_profile.insert(0, 'content_id', content_ids)

    if timer:
        print "User profile matrix time: ", (time() - start_time)

    pass recommend


'''
----------------------------------------------------------------------------
'''
if __name__ == '__main__':

    start_time = time()

    rmd = pd.read_csv('data/metadata/ResearchMetadata.csv')
    usa = pd.read_csv('data/metadata/User-SubjectArea.csv')
    rmd.loc[rmd.content_id == 18026, ['research_type']] = 0
    rmd.loc[rmd.content_id == 18027, ['research_type']] = 0
    rmd.loc[rmd.content_id == 18034, ['research_type']] = 0
    rmd.loc[rmd.content_id == 18324, ['research_type']] = 0
    rmd.loc[rmd.content_id == 18461, ['research_type']] = 0

    smd = pd.read_csv('data/modified_data/subject_metadata.csv')
    user_dlds = pd.read_csv('data/modified_data/top_downloads_users.csv')
    usd = pd.read_csv('data/modified_data/user-downloads.csv')
    fav = pd.read_csv('data/modified_data/user-favorites.csv')

    usr_mod = np.load('data/modified_data/users.npy')
    res_all = np.load('data/modified_data/content.npy')
    res_dld = np.load('data/modified_data/downloads.npy')

    utility = pd.read_csv('data/modified_data/utility_dld_matrix.csv')

    '''
    Construct item profile:
    '''

    practices = smd.practice.unique()
    areas = smd.SubjectAreaName.unique()

    mk = (rmd.research_type.unique() != rmd.research_type.unique()[10])
    types = rmd.research_type.unique()[mk]
    authors = rmd.author.unique()

    num_total_items = len(res_all)
    num_item_features = len(authors) + len(practices) + len(areas) + len(types)
    num_lim_features = len(practices) + len(areas)

    md_columns = ['practice', 'primary_subject_area', 'author', 'research_type']
    md_columns_lim = ['practice', 'primary_subject_area']

    features = np.append(np.append(np.append(practices, areas) , authors), types)
    features_lim = np.append(practices, areas)

    zero_data = np.zeros((num_total_items, num_item_features))
    zero_data_lim = np.zeros((num_total_items, num_lim_features))

    item_features = pd.DataFrame(zero_data, columns=features, dtype=int)
    item_features.insert(0, 'content_id', res_all)
    # item_features = populate_item_profile(item_features, rmd, md_columns)
    # item_features.to_csv('data/modified_data/item_features.csv',index=False)

    item_features_lim = pd.DataFrame(zero_data_lim, columns=features_lim, dtype=int)
    item_features_lim.insert(0, 'content_id', res_all)
    # item_features_lim = populate_item_profile(item_features_lim, rmd, md_columns_lim)
    # item_features_lim.to_csv('data/modified_data/item_features_small.csv',index=False)

    '''
    Construct user profile:
    '''

    num_total_users = len(usr_mod)

    # Get rid of Subject Area 21:
    masky = (usa.SubjectAreaID != 21)
    usa_mod = usa[masky]

    zero_data_users = np.zeros((num_total_users, num_item_features))
    zero_data_users_lim = np.zeros((num_total_users, num_lim_features))

    user_profiles = pd.DataFrame(zero_data_users, columns=features, dtype=int)
    user_profiles.insert(0, 'UserID', usr_mod)
    # user_profiles = populate_user_profile(user_profiles, utility, item_features)
    # user_profiles.to_csv('data/modified_data/user_profile',index=False)

    user_profiles.insert(0, 'UserID', usr_mod)
    utility.insert(0, 'UserID', usr_mod)
    item_features.insert(0, 'content_id', res_all)

    user_profiles_lim = pd.DataFrame(zero_data_users_lim, columns=features_lim, dtype=int)
    user_profiles_lim.insert(0, 'UserID', usr_mod)
    # user_profiles_lim = populate_base_user_profile(user_profiles_lim, smd, usa_mod)
    # user_profiles_lim.to_csv('data/modified_data/user_base_profile',index=False)

    print "Elapsed time: ", (time() - start_time)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as scs
import seaborn as sns
import random
from time import time

'''
Creates most basic recommender as a baseline test
'''

'''
    Recommend most favorited:
'''

def recommend_top(most_favs):
    '''
    Recommend most favorited content.

    INPUT:
    most_favs: (Pandas DataFrame) Sorted most favorited Content IDs

    OUTPUT:
    (int) Content ID of document
    '''

    return most_favs.content_id[0]

'''
    Recommend most favorited not yet downloaded:
'''

def recommend_most_fav(user_id, utility_matrix, most_favs):
    '''
    Recommend content by most favorite, not yet favorited by user.

    INPUT:
    user_id: (int or str)
    utility_matrix: (Pandas DataFrame) Utility matrix of all users and content
    most_favs: (Pandas DataFrame) Sorted most favorited Content IDs

    OUTPUT:
    recommend: (str) Content_ID of document
    '''

    user_id = str(user_id)
    user_favs = utility[utility.UserID == user_id]
    user_index = user_favs.index[0]
    content_index = 0
    recommend = str(most_favs.content_id[content_index])
    while user_favs.get_value(user_index,recommend):
        content_index += 1
        recommend = str(most_favs.content_id[content_index])

    return recommend

'''
    Recommend by random sampling from favorite distribution:
'''

def recommend_random_fav(dist, number):
    '''
    Recommend document by randomly sampling distribution of favorited docs.

    INPUT:
    dist: (Numpy array) History of favorites
    number: (int) Number of recommendations

    OUTPUT:
    (int) Content ID of document
    '''

    return np.random.choice(dist, size=number)

def plot_hist_basic(df, col):
    """
    Plot a histogram from the column col of dataframe df. Return a Matplotlib
    axis object.

    INPUT:
    df: (Pandas DataFrame)
    col: (str) Column from df with numeric data to be plotted

    OUTPUT:
    ax: (Matplotlib axis object)
    """
    data = df[col]
    ax = data.hist(bins=20, normed=1, edgecolor='none', figsize=(10,7), grid=False)
    ax.set_ylabel('Probability Density')
    ax.set_xlabel(col)

    return ax

def plot_kde(df, col):
    """
    Fit a Gaussian Kernal Density Estimate to some input data, plot the fit
    over a histogram of the data.
    INPUT:
    df: (Pandas DataFrame)
    col: (str) Column from df with numeric data to be plotted

    OUTPUT:
    ax: (Matplotlib axis object)
    """
    ax = plot_hist_basic(df, col)
    data = df[col]
    density = scs.kde.gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 100)
    kde_vals = density(x_vals)

    ax.plot(x_vals, kde_vals, 'r-')

    return ax

'''
    Recommend most recent, according to subject preference:
'''

def recent_subject(user, user_subjects, subject_areas, most_recent_items):
    '''
    Recommends most recent item in user's chosen subject area:

    INPUT:
    user: (int) User ID
    user_subjects: (pandas Dataframe) Users' chosen subject areas
    subject_areas: (pandas Dataframe) Subject area metadata
    most_recent_items: (pandas Dataframe) Most recent items by subject

    OUTPUT:
    rec: (str) Content ID of recommended document
    '''
    usa_mat = user_subjects.values

    mask = (usa_mat[:,0] == user)

    areas = np.array(zip(*usa_mat[mask])[1])

    area = np.random.choice(areas)

    area_n = subject_areas.loc[subject_areas.SubjectAreaID == area,\
        ['SubjectAreaName']].values

    if area_n:
        area_name = area_n.flatten()[0]

        reco = most_recent_items.loc[most_recent_items.primary_subject_area == \
            area_name, ['content_id']].values

        # print reco

        if reco:
            rec = reco.flatten()[0]
            # rec = most_recent_items.loc[most_recent_items.primary_subject_area == \
            #     area_name, ['content_id']].values.flatten()[0]

            return str(rec)
        else:
            return None
    else:
        return None


'''
Test baseline recommender:
'''

def score_baseline(utility, favorites):
    '''
    Calculates accuracy score of naive, random baseline recommender.

    INPUT:
    utility: (Pandas DataFrame) Utility matrix of users who have fav data
    favorites: (Numpy array) History of favorites

    OUTPUT:
    accuracy: (float) Accuracy score of recommender
    '''

    num = len(utility)
    preds = np.empty(num,dtype=int)

    recs = recommend_random_fav(favorites,num).astype(str)

    for i in xrange(num):
        preds[i] = utility.loc[i,recs[i]]

    accuracy = preds.mean()

    return accuracy

def score_most_recent(user_subjects, subject_areas, most_recent_items, utility):
    '''
    Calculates accuracy score of present recommender (most recent from subject).

    INPUT:
    user: (int) User ID
    user_subjects: (pandas Dataframe) Users' chosen subject areas
    subject_areas: (pandas Dataframe) Subject area metadata
    most_recent_items: (pandas Dataframe) Most recent items by subject
    utility: (Pandas DataFrame) Utility matrix of users who have dld data

    OUTPUT:
    score: (float) Accuracy score
    '''
    start_time = time()

    users = utility.UserID.values
    num_users = len(users)

    predictions = []
    count = 1
    for user in users:
        # print count, user

        rec = recent_subject(user, user_subjects, subject_areas, most_recent_items)

        if rec != None:
            val = utility.loc[utility.UserID == user, [rec]].values.flatten()[0]
        else:
            val = 0

        predictions.append(val)

        count += 1

    score = float(sum(predictions))/num_users

    print "Elapsed time: ", (time() - start_time)

    # return predictions
    return score, predictions

def score_most_recent_subset(user_subjects, subject_areas, most_recent_items,
    utility, users, both=False):
    '''
    Calculates accuracy score of present recommender (most recent from subject).

    INPUT:
    user: (int) User ID
    user_subjects: (pandas Dataframe) Users' chosen subject areas
    subject_areas: (pandas Dataframe) Subject area metadata
    most_recent_items: (pandas Dataframe) Most recent items by subject
    utility: (Pandas DataFrame) Utility matrix of users who have dld data
    users: (numpy array) Users of interest

    OUTPUT:
    score: (float) Accuracy score
    '''

    start_time = time()

    num_users = len(users)

    predictions = []
    count = 1
    for user in users:
        # print count, user

        rec = recent_subject(user, user_subjects, subject_areas, most_recent_items)

        if rec != None:
            val = utility.loc[utility.UserID == user, [rec]].values.flatten()[0]
        else:
            val = 0

        predictions.append(val)

        count += 1

    score = float(sum(predictions))/num_users

    print "Elapsed time: ", (time() - start_time)

    if both:
        return score, predictions
    else:
        return predictions

'''
------------------------------------------------------------------------------
'''
if __name__ == '__main__':
    start_time = time()

    plot = False

    # Import utility matrix:

    utility = pd.read_csv('data/modified_data/utility_matrix.csv') # OR:
    # utility = pd.read_csv('data/modified_data/utility_matrix.csv').drop(['Unnamed: 0'],axis=1)

    utility_dld = pd.read_csv('data/modified_data/utility_dld_matrix.csv')

    # fav = pd.read_csv('data/metadata/User-Research-Favorites.csv')
    fav = pd.read_csv('data/modified_data/user-favorites.csv')
    top_favs = fav.content_id.value_counts()

    fav_list = top_favs.to_frame().rename(columns={'content_id':'count'}). \
        reset_index().rename(columns={'index':'content_id'}).drop('count',1)

    #
    # X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 1)

    smd = pd.read_csv('data/modified_data/subject_metadata.csv')
    usa = pd.read_csv('data/metadata/User-SubjectArea.csv')

    # Distribution of favorited content ids:
    fav_dist = fav.content_id

    # Create utility matrix only for those who have favorited at least one doc:
    mask_favs = (utility.drop('UserID',1) > 0).any(axis=1)
    utility_favs = utility[mask_favs].reset_index(drop=True)

    # Create utility matrix only for those who have dowloaded at least one doc:
    mask_dlds = (utility_dld.drop('UserID',1) > 0).any(axis=1)
    utility_d = utility_dld[mask_dlds].reset_index(drop=True)


    # Create utility matrix only for those who have not favorited anything (newusrs):
    mask_new = (utility.drop('UserID',1) == 0).all(axis=1)
    utility_new = utility[mask_new].reset_index(drop=True)

    # Create utility matrix only for those who have not downloaded anything (newusrs):
    mask_d_new = (utility_dld.drop('UserID',1) == 0).all(axis=1)
    utility_d_new = utility_dld[mask_d_new].reset_index(drop=True)

    # Favorites distribution
    if plot:
        plot_kde(fav,'content_id')
        # plt.savefig('figures/research_faves_kde.png')
        plt.show()

    # Distribution of downloads by users:
    usd = pd.read_csv('data/modified_data/user-downloads.csv')
    top_dld_users = usd.UserID.value_counts()
    if plot:
        usd.UserID.hist(bins=100,grid=False)
        plt.xlabel('User ID')
        plt.ylabel('Count')
        plt.title('User Downloads')
        # plt.savefig('figures/userid_dld_hist.png')
        plt.show()

    # Save top download information
    user_dlds =  top_dld_users.to_frame().rename(columns={'UserID':'count'}). \
        reset_index().rename(columns={'index':'UserID'})
    user_dlds['Prop'] = user_dlds['count']/user_dlds['count'].sum()
    # user_dlds.to_csv('data/modified_data/top_downloads_users.csv',index=False)
    # user_dlds = pd.read_csv('data/modified_data/top_downloads_users.csv')

    user_dlds['count'].hist(grid=False,bins=300)
    if plot:
        plt.xlabel('Downloads/User')
        plt.ylabel('Count')
        plt.title('Downloads per User Distribution')
        # plt.savefig('figures/dlds_per_user_dist.png')
        plt.show()

        user_dlds['count'].hist(grid=False,bins=300,normed=True)
        plt.xlabel('Downloads/User')
        plt.ylabel('Count')
        plt.title('Downloads per User Distribution')
        # plt.savefig('figures/dlds_per_user_dist_norm.png')
        plt.show()

    tops = user_dlds[user_dlds['count'] > user_dlds['count'].mean()]
    print "Percent of downloads downloaded by top users (mean): ", tops.Prop.sum()
    if plot:
        tops['count'].hist(grid=False,bins=22)
        plt.xlabel('Downloads/User')
        plt.ylabel('Count')
        plt.title('Downloads/User, Most Active Users')
        # plt.savefig('figures/dlds_per_user_dist_most.png')
        plt.show()

        tops[tops['count'] > 500]['count'].hist(grid=False)
        plt.xlabel('Downloads/User')
        plt.ylabel('Count')
        plt.title('Downloads/User, Most Active Users (>500)')
        # plt.savefig('figures/dlds_per_user_dist_most_500.png')
        plt.show()


    prop = user_dlds.Prop.values
    index = 100
    print "The top {} users represent {} of the download data.".format(index,
        np.sum(prop[0:index]))
    index = 500
    print "The top {} users represent {} of the download data.".format(index,
        np.sum(prop[0:index]))

    # Research by subject area:

    rmd = pd.read_csv('data/metadata/ResearchMetadata.csv')
    rmd_mod = rmd[['content_id','publish_date','primary_subject_area']].copy()
    recent_subj_id = rmd_mod.groupby(['primary_subject_area'],as_index=False).max()
    # recent_subj_id.to_csv('data/modified_data/recent_by_subject.csv')

    # Scores:

    top_fav_score = float(top_favs.max())/len(utility_favs)
    random_fav_score = score_baseline(utility_favs,fav_dist)
    # score, preds = score_most_recent(usa,smd,recent_subj_id,utility_dld)
    ''' Takes too long, score is: '''
    score = 0.0071465

    cutoff = 500
    top_users = user_dlds.UserID.values[0:cutoff]
    top_score, top_predicitons = score_most_recent_subset(usa,smd,recent_subj_id,
        utility_dld,top_users)


    print "The most naive baselines are:"
    print "Recommending top favorite: ", top_fav_score
    print "Recommending random favorite: ", random_fav_score
    print "Recommending most recent: ", score
    print "Recommending most rect to top users: ", top_score

    end_time = time()
    print "Time elapsed: ", end_time - start_time

    # '''How to find index of elements of an array inside another array:'''
    # x = np.array([3,5,7,1,9,8,6,6])
    # y = np.array([2,1,5,10,100,6])
    # 
    # index = np.argsort(x)
    # sorted_x = x[index]
    # sorted_index = np.searchsorted(sorted_x, y)
    #
    # yindex = np.take(index, sorted_index, mode="clip")
    # mask = x[yindex] != y
    #
    # result = np.ma.array(yindex, mask=mask)
    # print result

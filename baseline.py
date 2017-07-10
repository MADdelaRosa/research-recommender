import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as scs
import seaborn as sns
import random

'''
Creates most basic recommender as a baseline test
'''

# Import utility matrix:

utility = pd.read_csv('data/utility_matrix.csv').drop(['Unnamed: 0'],axis=1)
fav = pd.read_csv('data/metadata/User-Research-Favorites.csv')
top_favs = fav.content_id.value_counts()

fav_list = top_favs.to_frame().rename(columns={'content_id':'count'}). \
    reset_index().rename(columns={'index':'content_id'}).drop('count',1)

#
# X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 1)

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

# Distribution of favorited content ids:
fav_dist = fav.content_id

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

# Create utility matrix only for those who have favorited at least one doc:
mask = (utility.drop('UserID',1) > 0).any(axis=1)
utility_favs = utility[mask].reset_index(drop=True)

# Create utility matrix only for those who have not favorited anything (newusrs):
mask = (utility.drop('UserID',1) == 0).all(axis=1)
utility_new = utility[mask].reset_index(drop=True)

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

plot_kde(fav,'content_id')
# plt.savefig('figures/research_faves_kde.png')
plt.show()


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

top_fav_score = float(top_favs.max())/len(utility_favs)
random_fav_score = score_baseline(utility_favs,fav_dist)

print "The most naive baselines are:"
print "Recommending top favorite: ", top_fav_score
print "Recommending random favorite: ", random_fav_score

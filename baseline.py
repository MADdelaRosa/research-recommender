import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

'''
Creates most basic recommender as a baseline test
'''

# Import utility matrix:

utility = pd.read_csv('data/utility_matrix.csv')
fav = pd.read_csv('data/metadata/User-Research-Favorites.csv')
top_favs = fav.content_id.value_counts()

fav_list = top_favs.to_frame().rename(columns={'content_id':'count'}). \
    reset_index().rename(columns={'index':'content_id'}).drop('count',1)


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 1)


def recommend_most_fav(user_id, utility_matrix, most_favs):

    user_id = str(user_id)
    user_favs = utility[utility.UserID == user_id]
    user_index = user_favs.index[0]
    content_index = 0
    recommend = str(most_favs.content_id[content_index])
    while user_favs.get_value(user_index,recommend):
        content_index += 1
        recommend = str(most_favs.content_id[content_index])

    return recommend

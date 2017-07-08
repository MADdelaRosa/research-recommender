import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

'''
Creates most basic recommender as a baseline test
'''

# Import utility matrix:

fav = pd.read_csv('data/utility_matrix.csv')


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 1)


def recommend_most_fav(user_id, utility_matrix, most_favs):
    

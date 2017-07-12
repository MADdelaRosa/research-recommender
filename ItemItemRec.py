import pandas as pd
import numpy as np
import time as time
from __future__ import division
from collaborative_filtering import jaccard_dist, cos_dist

class ItemItemRec(object):

    def __init__(self, n_size):
        self.n_size = n_size

    def fit(self, utility_matrix):
        self.utility_matrix = utility_matrix
        self.n_users = utility_matrix.shape[0]
        self.n_items = utility_matrix.shape[1]

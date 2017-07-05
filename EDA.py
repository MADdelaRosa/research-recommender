import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download data
dld = pd.read_csv('data/bersin_download_data_2007-2017.csv')

# Metadata
rmd = pd.read_csv('data/metadata/ResearchMetadata.csv')
smd = pd.read_csv('data/metadata/SubjectArea-Metadata.csv')
fav = pd.read_csv('data/metadata/User-Research-Favorites.csv')
rat = pd.read_csv('data/metadata/User-Research-Rating.csv')
usa = pd.read_csv('data/metadata/User-SubjectArea.csv')
usr = pd.read_csv('data/metadata/Users.csv')
usd = pd.read_csv('data/metadata/User-Downloads.csv')
uscom = pd.read_csv('data/metadata/User-Comments.csv')
uscon = pd.read_csv('data/metadata/User-Conversations.csv')


# Dataframe of only paid member accounts:
#dld.user_account_status.value_counts()
dld_paid = dld[dld.user_account_status == 'paid_member']
dld_comp = dld[dld.user_account_status == 'complimentary_account']

# filter out paid members that are within the company:
mask = dld_paid[dld_paid['email'].str.contains('deloitte')]
dld_ext = dld_paid.drop(mask.index)

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

# Explore download history:
# Download histogram
dld_ext.research_id.hist(grid=False)
plt.xlabel('Research ID')
plt.ylabel('Count')
#plt.show()
# Most downloaded:
histos = dld_ext.research_id.value_counts()
id_most = histos.index[0]
ind_most = rmd[rmd.content_id == id_most].index[0]
print 'Most downloaded research_id: ', id_most
print 'Title: ', rmd.content_title[ind_most]
print 'Long description: ', rmd.long_description[ind_most]

def most_downloaded(download_df, research_df, number):
    counts = download_df.research_id.value_counts()
    for i in xrange(number):
        res_id = int(counts.index[i])
        ind = research_df[research_df.content_id == res_id].index[0]
        print 'The number {} most downloaded research_id: {}'.format(i+1, res_id)
        print 'Title: ', research_df.content_title[ind]
        print 'Author: ', research_df.author_ordered[ind]
        print 'Long description: ', research_df.long_description[ind]
        print '\n'

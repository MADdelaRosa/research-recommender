import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Organize data:
'''

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

'''
Explore download history:
'''

# Download histogram:
dld_ext.research_id.hist(grid=False,bins=100)
plt.xlabel('Research ID')
plt.ylabel('Count')
plt.savefig('figures/res_id_hist.png')
plt.show()

# Most downloaded:
histos = dld_ext.research_id.value_counts()
id_most = histos.index[0]
num_dlds = histos[histos.index[0]]
ind_most = rmd[rmd.content_id == id_most].index[0]
print 'Most downloaded research_id: ', id_most
print 'It was downloaded {} times.'.format(num_dlds)
print 'Title: ', rmd.content_title[ind_most]
print 'Long description: ', rmd.long_description[ind_most]

def most_downloaded(download_df, research_df, number):
    counts = download_df.research_id.value_counts()
    for i in xrange(number):
        res_id = int(counts.index[i])
        num_downloads = count[res_id]
        ind = research_df[research_df.content_id == res_id].index[0]
        print 'The number {} most downloaded research_id: {}'.format(i+1, res_id)
        print 'It was downloaded {} times.'.format(num_downloads)
        print 'Title: ', research_df.content_title[ind]
        print 'Author: ', research_df.author_ordered[ind]
        print 'Long description: ', research_df.long_description[ind]
        print '\n'

'''
Explore redundant download data:
'''

print 'Number of unique download dates: ', dld_ext.download_date.unique().size
print 'Out of {} total download dates.'.format(dld_ext.shape[0])

# Repeated dates:
repdates = dld_ext.download_date.value_counts()

'''
Favorites data:
'''

fav.UserID.hist(grid=False,bins=100)
plt.xlabel('User ID')
plt.ylabel('Count')
plt.savefig('figures/user_faves.png')
plt.show()

fav.content_id.hist(grid=False,bins=100)
plt.xlabel('Content ID')
plt.ylabel('Count')
plt.savefig('figures/research_faves.png')
plt.show()

# Plot downloads/favorites together:

# Downloads from usd:
usd.content_id.hist(grid=False,bins=100)
plt.xlabel('Content ID')
plt.ylabel('Count')
plt.savefig('figures/content_downloads_usd.png')
plt.show()

usd.content_id.hist(grid=False,bins=100)
fav.content_id.hist(grid=False,bins=100)
plt.xlabel('Content ID')
plt.ylabel('Count')
plt.savefig('figures/research_dlds_faves.png')
plt.show()

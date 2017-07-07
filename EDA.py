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
sol = pd.read_csv('data/metadata/Solution-Providers.csv')

# This is a mess:
allres = pd.read_csv('data/metadata/AllResearch-Incl-Purged.csv')


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
# dld_ext.research_id.hist(grid=False,bins=100)
# plt.xlabel('Research ID')
# plt.ylabel('Count')
# plt.savefig('figures/res_id_hist.png')
# plt.show()

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
        num_downloads = counts[res_id]
        ind = research_df[research_df.content_id == res_id].index[0]
        print 'The number {} most downloaded research_id: {}'.format(i+1, res_id)
        print 'It was downloaded {} times.'.format(num_downloads)
        print 'Title: ', research_df.content_title[ind]
        print 'Author: ', research_df.author[ind]
        print 'Long description: ', research_df.long_description[ind]
        print '\n'

'''
Explore redundant download data:
'''

print 'Number of unique download dates: ', dld_ext.download_date.unique().size
print 'Out of {} total download dates.'.format(dld_ext.shape[0])

# Repeated dates:

'''
Favorites data:
'''

# fav.UserID.hist(grid=False,bins=100)
# plt.xlabel('User ID')
# plt.ylabel('Count')
# plt.savefig('figures/user_faves.png')
# plt.show()
#
# fav.content_id.hist(grid=False,bins=100)
# plt.xlabel('Content ID')
# plt.ylabel('Count')
# plt.savefig('figures/research_faves.png')
# plt.show()

# Plot downloads/favorites together:

# Downloads from usd:
# usd.content_id.hist(grid=False,bins=100)
# plt.xlabel('Content ID')
# plt.ylabel('Count')
# plt.savefig('figures/content_downloads_usd.png')
# plt.show()
#
usd.content_id.hist(grid=False,bins=100,normed=True)
fav.content_id.hist(grid=False,bins=100,normed=True)
plt.xlabel('Content ID')
plt.ylabel('Count')
plt.savefig('figures/research_dlds_faves.png')
plt.show()

'''
Find documents with no metadata:
'''

# Research content downloaded (unique):
res_id_dld = usd.content_id.unique()
# Research content with available metadata:
res_id_rmd = rmd.content_id.unique()
# Research content favorited (unique):
res_id_fav = fav.content_id.unique()
# Compare sizes:
print "Same amount of research metadata as downloaded documents?"
print res_id_dld.size == res_id_rmd.size

def filter_out_short_from_large(array1,array2):
    '''
    Finds and filters out elements not in common between two arrays
    '''
    n1 = len(array1)
    n2 = len(array2)
    if n1 > n2:
        nlarge = n1
        nshort = n2
        alarge = array1
        ashort = array2
    else:
        nlarge = n2
        nshort = n1
        alarge = array2
        ashort = array1

    mask = np.empty(nlarge, dtype=bool)
    # print (mask == True).any()
    for i in xrange(nshort):
        tarr = alarge == ashort[i]
        if (tarr == True).any():
            mask[int(np.where(tarr == True)[0])] = True
        # print mask[-1]
        # if mask[-1] == True:
        #     print i
        #     print ashort[i]
        #     # print (tarr == True).any()
        #     break

    return mask, alarge[mask]

def filter_out_uncommon_elements(array1,array2):
    '''
    Finds and filters out elements in array1 not found in array2
    '''
    n1 = len(array1)
    n2 = len(array2)

    mask = np.empty(n1, dtype=bool)

    for i in xrange(n2):
        tarr = array1 == array2[i]
        if (tarr == True).any():
            mask[int(np.where(tarr == True)[0])] = True

    return mask, array1[mask]

# mask1, res_dld = filter_out_uncommon_elements(res_id_dld,res_id_rmd)
# mask2, res_fav = filter_out_uncommon_elements(res_id_fav,res_id_rmd)
# print "Length of mask1: ", mask1.shape
# print "Length of res_dld: ", res_dld.shape
# print "Length of mask2: ", mask2.shape
# print "Lenght of res_fav: ", res_fav.shape

'''
That absolutely did not work
'''

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
dld_ext.research_id.hist(grid=False,bins=100)
plt.xlabel('Research ID')
plt.ylabel('Count')
# plt.savefig('figures/res_id_hist.png')
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

fav.UserID.hist(grid=False,bins=100)
plt.xlabel('User ID')
plt.ylabel('Count')
# plt.savefig('figures/user_faves.png')
plt.show()

fav.content_id.hist(grid=False,bins=100)
plt.xlabel('Content ID')
plt.ylabel('Count')
# plt.savefig('figures/research_faves.png')
plt.show()

# Plot downloads/favorites together:

# Downloads from usd:
usd.content_id.hist(grid=False,bins=100)
plt.xlabel('Content ID')
plt.ylabel('Count')
# plt.savefig('figures/content_downloads_usd.png')
plt.show()
#
usd.content_id.hist(grid=False,bins=100,normed=True)
fav.content_id.hist(grid=False,bins=100,normed=True)
plt.xlabel('Content ID')
plt.ylabel('Count')
# plt.savefig('figures/research_dlds_faves.png')
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

    msk = np.empty(nlarge, dtype=bool)
    # print (mask == True).any()
    for i in xrange(nshort):
        tarr = alarge == ashort[i]
        if (tarr == True).any():
            msk[int(np.where(tarr == True)[0])] = True
        # print mask[-1]
        # if mask[-1] == True:
        #     print i
        #     print ashort[i]
        #     # print (tarr == True).any()
        #     break

    return msk, alarge[msk]

def filter_out_uncommon_elements(array1,array2):
    '''
    Finds and filters out elements in array1 not found in array2
    '''
    n1 = len(array1)
    n2 = len(array2)

    msk = np.empty(n1, dtype=bool)

    for i in xrange(n2):
        tarr = array1 == array2[i]
        if (tarr == True).any():
            msk[int(np.where(tarr == True)[0])] = True

    return msk, array1[msk]

# mask1, res_dld = filter_out_uncommon_elements(res_id_dld,res_id_rmd)
# mask2, res_fav = filter_out_uncommon_elements(res_id_fav,res_id_rmd)
# print "Length of mask1: ", mask1.shape
# print "Length of res_dld: ", res_dld.shape
# print "Length of mask2: ", mask2.shape
# print "Lenght of res_fav: ", res_fav.shape

'''
That absolutely did not work
'''

'''
Use python sets on:
# Research content downloaded (unique):
res_id_dld
# Research content with available metadata:
res_id_rmd
# Research content favorited (unique):
res_id_fav
(can also use np.setdiff1d(a,b) but less explicit)
'''

D = set(res_id_dld)
R = set(res_id_rmd)
F = set(res_id_fav)

'''
Total downloads and research metadata:
'''
# TD is the total number of documents, the union of downloads and reserach mdata:
TD = R | D

# V is the set of documents with existing mdata, not downloaded:
V = TD - D

# W is the set of documents downloaded yet no mdata exists for them
W = TD - R

# C is the intersection of D and R, documents downloaded with mdata:
C = D & R #(C = TD - V - W)

'''
Total favorited and research metadata:
'''
# TF is the total favorited and with mdata:
TF = R | F

# N is the set of documents with mdata but not favorited:
N = TF - F

# M is the set of documents favorited for which we have no mdata:
M = TF - R

# G is the intersection of R and F, documents favorited with mdata:
G = F & R #(G = TF - M - N)

'''
Total favorited and downloaded:
'''
# FD is the total favorited and downloaded:
FD = F | D

# P is the set of documents downloaded but not favorited:
P = FD - F

# Q is the set of documents favorited but not downloaded (hopefully empty):
Q = FD - D  #(Note: this is the empty set)

# J is the intersection of F and D, documents downloaded and favorited:
J = F & D  #(Note: this set is F itself)

res_dl = np.sort(np.array(list(C)))

res_all = np.sort(np.array(list(TD)))

'''
Create Utility Matrix:
'''

# All users:
all_usr = usr.UserID.unique()
res_str = res_all.astype(str)
all_usr = all_usr.astype(str)

# Initialize the matrix:
zero_data = np.zeros((all_usr.size,res_str.size))
# utility = pd.DataFrame(zero_data, index=all_usr, columns=res_all, dtype=int)
utility = pd.DataFrame(zero_data, columns=res_str, dtype=int)
utility.insert(0, 'UserID', all_usr)

for index in xrange(len(fav)):
    user_id = str(fav.UserID[index])
    content_id = str(fav.content_id[index])
    utility.loc[utility['UserID'] == user_id, [content_id]] = 1

utility.to_csv('data/utility_matrix.csv') # OR:
# utility.to_csv('data/utility_matrix.csv', index=False)

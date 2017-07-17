import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

'''
Organize data:
'''

# Download data
dld = pd.read_csv('data/bersin_download_data_2007-2017.csv').drop(['Unnamed: 0'], axis=1)

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
# allres = pd.read_csv('data/metadata/AllResearch-Incl-Purged.csv')


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
#
# fav.UserID.hist(grid=False,bins=100)
# plt.xlabel('User ID')
# plt.ylabel('Count')
# # plt.savefig('figures/user_faves.png')
# plt.show()
#
# fav.content_id.hist(grid=False,bins=100)
# plt.xlabel('Content ID')
# plt.ylabel('Count')
# # plt.savefig('figures/research_faves.png')
# plt.show()
#
# # Plot downloads/favorites together:
#
# # Downloads from usd:
# usd.content_id.hist(grid=False,bins=100)
# plt.xlabel('Content ID')
# plt.ylabel('Count')
# # plt.savefig('figures/content_downloads_usd.png')
# plt.show()
# #
# usd.content_id.hist(grid=False,bins=100,normed=True)
# fav.content_id.hist(grid=False,bins=100,normed=True)
# plt.xlabel('Content ID')
# plt.ylabel('Count')
# # plt.savefig('figures/research_dlds_faves.png')
# plt.show()

'''
Research document inconsistencies.
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

mask1, res_dld = filter_out_uncommon_elements(res_id_dld,res_id_rmd)
mask2, res_fav = filter_out_uncommon_elements(res_id_fav,res_id_rmd)
print "Length of mask1: ", mask1.shape
print "Length of res_dld: ", res_dld.shape
print "Length of mask2: ", mask2.shape
print "Lenght of res_fav: ", res_fav.shape

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
# TD is the total number of documents, the union of downloads and research mdata:
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

# np.save('data/modified_data/downloads.npy',res_dl)
# np.save('data/modified_data/content.npy',res_all)


'''
User ID inconsistencies.
Extract User IDs from all tables:
'''

# User (unique):
us_id_usr = usr.UserID.unique()
# Users from user download data (unique):
us_id_usd = usd.UserID.unique()
# Users from user favorite data (unique):
us_id_fav = fav.UserID.unique()
# Users from user metadata (unique):
us_id_usa = usa.UserID.unique()

UU = set(us_id_usr)
UD = set(us_id_usd)
UF = set(us_id_fav)
UA = set(us_id_usa)

'''
    Users from Favorites and Downloads:
'''
# TDF is total users who favorite and downloaded at least once:
TDF = UD | UF

# A is the set of all users who downloaded without favoriting:
A = TDF - UF

# B is the set of all users who favorited but not downloaded (odd):
B = TDF - UD

# FUD is the intersection of UD and UF, users who both downloaded and favorited:
FUD = UD & UF

# Surprisingly, there are 49 users (B) who favorited w/o downloading
# Use TDF going forward

'''
    Users from Favorites and Downloads plus Users Table:
'''

# TOT is the total number of users, fav/dld plus those who don't:
TOT = UU | TDF

# R is the set of all users did not dld or fav anything:
R = TOT - TDF

# S is the set of all users NOT found in the User table (SHOULD BE EMPTY):
S = TOT - UU

# There are 1061 Users not in the User Table (a big problem)
# Whence?:

MD = (UU | UD) - UU # 1059 users here
MF = (UU | UF) - UU # There's the 70 missing users

print "Check they are all in set S: "
print (MD | MF) == S # Out: True

# Therefore, drop users in S from downloads (usd) and favorites (fav)
# drop_usr = np.sort(np.array(list(S)))

'''
    Users from User Subject Area:
'''

# TAU is the total number of users in Users and Subject Area tables:
TAU = UU | UA

# X is the set of all users with no Subject Area choice
X = TAU - UA

# Y is the set of all users with Subject Area choice but no User profile
Y = TAU - UU

print "There are {} users with no chosen Subject Area".format(len(X))
print "There are {} users who chose a Subject Area but have no profile".format(len(Y))

# The union of these two sets is a candidate to be dropped:
Z = X | Y

# However, set S (which must be dropped from user list) is not a subset of X U Y:
print "Is set of dld/fav users no it table a subset of Users and SubjectArea mismatch?:"
print S <= Z

# How many user favorites will be lost:
print "There are {} users in fav table that are in the User/SubjectArea mismatch.".format(len(UF & Z))

# How many user downloads will be lost:
print "There are {} users in dld table that is in the User/SubjectArea mismatch.".format(len(UD & Z))

# So, we drop the union of S and Z, which encompasses all the mismatches with Users table:
MM = S | Z

drop_usr = np.sort(np.array(list(MM)))

'''
Subject Area Metadata incomplete:
'''

# Subject Area Metadata (smd) is missing 11 subject areas
# Users have choses all 51 except 48 and 49 (skipped)
# Obtained information for 3, 4, 9, 10, 14, 18, 20, and 22:

# Create array to be inserted into smd:

'''
3 to 7 - Learning & Development [4]
4 to 26 Talent Acquisition [16]
9 to 7 Learning & Development [4]
10 to 24 Talent Acquisition [14]
14 to 15 Talent Management [9]
18 to 41 Human Resources [31]
20 to 41 Human Resources [31]
22 to 51 Tools & Technology [39]
'''

# Move SubjectAreaID to index:
smd.set_index('SubjectAreaID',inplace=True)

# Add new rows:
smd.loc[3] = [smd.iat[4,0], smd.iat[4,1]]
smd.loc[4] = [smd.iat[16,0], smd.iat[16,1]]
smd.loc[9] = [smd.iat[4,0], smd.iat[4,1]]
smd.loc[10] = [smd.iat[14,0], smd.iat[14,1]]
smd.loc[14] = [smd.iat[9,0], smd.iat[9,1]]
smd.loc[18] = [smd.iat[31,0], smd.iat[31,1]]
smd.loc[20] = [smd.iat[31,0], smd.iat[31,1]]
smd.loc[22] = [smd.iat[39,0], smd.iat[39,1]]

smd.sort_index(inplace=True)

smd.reset_index(inplace=True)

# smd.to_csv('data/modified_data/subject_metadata.csv',index=False)

# Area 21 is not a real Subject Area, so we must drop users who ONLY selected it:

sa21 = usa[usa.SubjectAreaID == 21].UserID

print "There are {} users who selected SA 21.".format(len(sa21))

# Frequency counts for choice of Subject Area:
usr_subs = usa.UserID.value_counts()

def drop_usr_sa21(sa21_array, sa_freqs):
    '''
    Find the UserIDs of users who chose ONLY subject area 21:

    INPUT:
    sa21_array: (numpy array) array of users who chose SA 21
    sa_freqs: (numpy array) array of UserID freqs in user subject area dataframe

    OUTPUT:
    users_drop: (numpy array) array of UserIDs to drop
    '''
    users_drop = []

    for value in sa21_array:
        if sa_freqs[value] == 1:
            users_drop.append(value)

    users_drop = np.asarray(users_drop)

    if users_drop:
        return users_drop
    else:
        return None

drop_usr_sa21(sa21,usr_subs) # Returns None object

'''
Drop all mismatched User IDs from fav, usd using drop_usr:
'''

def drop_usr_df(df, users):
    '''
    Drops UserIDs from dataframe.

    INPUT:
    df: (pandas DataFrame) Dataframe to drop IDs from
    users: (numpy array) User IDs to drop

    OUTPUT:
    df: (pandas DataFrame) Modified DataFrame
    '''
    start_time = time()

    # Reduce the number of iterations:
    U = set(df.UserID)
    V = set(users)
    common_users = np.sort(np.array(list(U & V)))

    for value in common_users:
        df = df[df.UserID != value]

    df.reset_index(inplace=True)

    end_time = time()
    print "Time elapsed: ", end_time - start_time

    return df

# new_fav = drop_usr_df(fav,drop_usr)
# new_usd = drop_usr_df(usd,drop_usr)

# new_fav.to_csv('data/modified_data/user-favorites.csv', index=False)
# new_usd.to_csv('data/modified_data/user-downloads.csv', index=False)

'''
Create Utility Matrix:
'''

# All users:
# all_usr = usr.UserID.unique()
#
# usr_mod = np.sort(np.setdiff1d(all_usr,drop_usr))

usr_mod =  np.sort(np.array(list(UU - MM)))
# np.save('data/modified_data/users.npy',usr_mod)

# Recast into string
# res_str = res_all.astype(str)
# all_usr = all_usr.astype(str)

# Drop users found in drop_usr from fav and dld

# # Initialize the matrix:
# zero_data = np.zeros((all_usr.size,res_str.size))
# # utility = pd.DataFrame(zero_data, index=all_usr, columns=res_all, dtype=int)
# utility = pd.DataFrame(zero_data, columns=res_str, dtype=int)
# utility.insert(0, 'UserID', all_usr)
#
# for index in xrange(len(fav)):
#     user_id = str(fav.UserID[index])
#     content_id = str(fav.content_id[index])
#     utility.loc[utility['UserID'] == user_id, [content_id]] = 1

def make_utility_matrix(users, content, favorites):
    '''
    Constructs utility matrix from content and user favorites data

    INPUT:
    users: (numpy array) UserIDs of all users
    content: (numpy array) ContentIDs of all content
    favorites: (pandas DataFrame) Data with all instances of user favoriting

    OUTPUT:
    utility_matrix: (pandas DataFrame) User/Content unitlity matrix
    '''
    start_time = time()

    zero_data = np.zeros((users.size,content.size))

    utility_matrix = pd.DataFrame(zero_data, columns=content, dtype=int)
    utility_matrix.insert(0, 'UserID', users)

    for index in xrange(len(favorites)):
        user_id = favorites.UserID[index]
        content_id = favorites.content_id[index]
        utility_matrix.loc[utility_matrix['UserID'] == user_id, [content_id]] = 1

    end_time = time()
    print "Time elapsed: ", end_time - start_time

    return utility_matrix

# utility = make_utility_matrix(usr_mod, res_all, fav)
# utility_dld = make_utility_matrix(usr_mod, res_all, usd)

## utility.to_csv('data/modified_data/utility_matrix.csv') # OR:
# utility.to_csv('data/modified_data/utility_matrix.csv', index=False)
# utility_dld.to_csv('data/modified_data/utility_dld_matrix.csv', index=False)

import pandas as pd
import numpy as np

'''
Create master donwload table:
'''

d1 = pd.read_csv('data/DownloadDataBefore2014.csv')
d2 = pd.read_csv('data/DownloadDataAfter2014.csv')

d_1 = pd.read_csv('data/Bersin download data-2007-2013.csv')
d_2 = pd.read_csv('data/Bersin download data-2014-2017.csv')

'''
Concatenate tables:
'''

dld = pd.concat([d1,d2], ignore_index=True)
dld_old = pd.concat([d_1,d_2], ignore_index=True)

'''
Save dataframes as csv files:
'''

# dld.to_csv('data/download_data.csv', index=False)
# dld_old.to_csv('data/bersin_download_data.csv', index=False)

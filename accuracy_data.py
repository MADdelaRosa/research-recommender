import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''
Accuracy Data:
'''
'''
                            1         3      5     10       50    100'''
acc_500_10_0  = np.array([0.014, 0.044,    0.05,  0.09,  0.184, 0.256])
acc_500_50_0  = np.array([0.034, 0.044,   0.034,  0.09,  0.178, 0.246])
acc_500_100_0 = np.array([0.02,  0.047, 0.07425, 0.074,  0.176, 0.258])

acc_500_10_1  = np.array([0.006, 0.026,  0.038,  0.054, 0.122, 0.19])
acc_500_50_1  = np.array([0.01,  0.028, 0.034,   0.046, 0.106, 0.138])
acc_500_100_1 = np.array([0.012, 0.014,  0.042,  0.048, 0.128, 0.144])

'''
                            1   3   5   10  50  100 '''
acc_1500_10_0  = np.array([0,  0,  0,  0,  0, 0])
acc_1500_50_0  = np.array([0,  0,  0,  0.0666,  0, 0])
acc_1500_100_0 = np.array([0,  0,  0,  0.06533,  0.12733, 0])

acc_1500_10_1  = np.array([0.016,  0,  0,  0,  0, 0])
acc_1500_50_1  = np.array([0.01,  0,  0,  0.052,  0, 0])
acc_1500_100_1 = np.array([0,  0,  0.0531,  0.04933,  0.11266, 0])

'''
Build results matrix:
'''

zerodata = np.zeros((36,4))

N = [10,50,100]
recs = [1,3,5,10,50,100]

zerodata[18:36,0] = 1

numrecs = len(recs)

for i in xrange(numrecs):
    zerodata[i,2] = recs[i]
    zerodata[i,3] = acc_500_10_0[i]

    zerodata[i+numrecs,2] = recs[i]
    zerodata[i+numrecs,3] = acc_500_50_0[i]

    zerodata[i+2*numrecs,2] = recs[i]
    zerodata[i+2*numrecs,3] = acc_500_100_0[i]

    zerodata[i+3*numrecs,2] = recs[i]
    zerodata[i+3*numrecs,3] = acc_500_10_1[i]

    zerodata[i+4*numrecs,2] = recs[i]
    zerodata[i+4*numrecs,3] = acc_500_50_1[i]

    zerodata[i+5*numrecs,2] = recs[i]
    zerodata[i+5*numrecs,3] = acc_500_100_1[i]

numn = len(N)

# zerodata[0:6,1] = N[0]
# zerodata[18:24,1] = N[0]
# zerodata[6:12,1] = N[1]
# zerodata[24:30,1] = N[1]
# zerodata[12:18,1] = N[2]
# zerodata[30:36,1] = N[2]

for i in xrange(numn):
    zerodata[i*numrecs:(i+1)*numrecs,1] = N[i]
    zerodata[numn*numrecs+(i*numrecs):numn*numrecs+((i+1)*numrecs),1] = N[i]

cols = ['Breadth', 'N-size', 'recs', 'accuracy']

acc_500 = pd.DataFrame(zerodata, columns=cols)

x = acc_500.iloc[0:6,2].values
y = acc_500.iloc[0:6,3].values
plt.plot(x,y,'b.-', label='N=10,Breadth')
x = acc_500.iloc[6:12,2].values
y = acc_500.iloc[6:12,3].values
plt.plot(x,y,'g.-', label='N=50,Breadth')
x = acc_500.iloc[12:18,2].values
y = acc_500.iloc[12:18,3].values
plt.plot(x,y,'r.-', label='N=100,Breadth')
x = acc_500.iloc[18:24,2].values
y = acc_500.iloc[18:24,3].values
plt.plot(x,y,'b*--', label='N=10,Depth')
x = acc_500.iloc[24:30,2].values
y = acc_500.iloc[24:30,3].values
plt.plot(x,y,'g*--', label='N=50,Depth')
x = acc_500.iloc[30:36,2].values
y = acc_500.iloc[30:36,3].values
plt.plot(x,y,'r*--', label='N=100,Depth')
plt.xlabel('Number of Recommendations')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('figures/acc_500.png')
plt.close()

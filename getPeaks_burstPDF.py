# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:44:57 2020

@author: katic
"""
from unidip import UniDip
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture as GM
from sklearn.neighbors import KernelDensity as KD
import matplotlib.pyplot as plt
import get_kinematics as kin

#def movingave(a, n=1) :
#    ret = np.cumsum(a, dtype=float)
#    ret[n:] = ret[n:] - ret[:-n]
#    return ret[n - 1:] / n

nbins = 90

pathname = r'C:\Users\katic\OneDrive\Documents\MPI\FishVR\data\forDebug'
session = 'test_data.csv.gz'
fishdata = pd.read_csv(pathname+'\\'+session)

ind_noVF = np.arange(0,np.where(fishdata.osg_fish1_y !=0)[0][0]) #if only pre-stimulus data is desired
fishdata = fishdata.iloc[ind_noVF]

pos= np.empty([fishdata.shape[0],3])
pos[:,0:3] = fishdata[['x','y','z']].to_numpy()

time = np.empty(fishdata.shape[0])
time[:] = fishdata.framenumber.to_numpy() - fishdata.loc[0,'framenumber0']
time = time * 0.01

ori = kin.getOri(pos)
speed = kin.getSpeed(pos,time)

newvals  = kin.cleanData(pos = pos[1:], ori = ori,\
                      speed = speed, time=time[1:],\
                      doSmooth = True, nbinsToAve = 5 )

bursts,burstrate= kin.getBGcycle(speed,time,burstHtthresh = 0.0,threshold = 0.01)

plt.figure()
bursts.burstHt.hist(bins=120)


#normal mixtures
bspeeds = bursts.burstHt.values.reshape(-1,1)
nparts = 2
model = GM(nparts).fit(bspeeds)
aic = model.aic(bspeeds)
bic = model.bic(bspeeds)
peakmeans = model.means_
peakvar = model.covariances_

fig = plt.figure(figsize=(8, 8))
x = np.linspace(0, .20, 1000)
logprob = model.score_samples(x.reshape(-1, 1))
responsibilities = model.predict_proba(x.reshape(-1, 1))
pdf = np.exp(logprob)
pdf_individual = responsibilities * pdf[:, np.newaxis]
plt.hist(bspeeds, 30, density=True, histtype='stepfilled', alpha=0.4)
plt.plot(x, pdf, '-k')
plt.plot(x, pdf_individual, '--k')
ax = plt.gca()
ax.set_xlabel('$speed$')
ax.set_ylabel('$p(speed)$')


#kernel density estimates
kdmodel = KD(bandwidth = .008, kernel = "gaussian")
sample = bspeeds
kdmodel.fit(sample)
values = x
values = values.reshape((len(values), 1))
probabilities = kdmodel.score_samples(values)
probabilities = np.exp(probabilities)
# plot the histogram and pdf
fig = plt.figure(figsize=(8, 8))
plt.hist(sample, bins=50, density=True, alpha = 0.5)
plt.plot(values[:], probabilities)

ind = np.gradient(np.sign(np.gradient(probabilities))) == -1
aux = values[ind]
threshold = 0.02
newpeak = np.ones(aux.shape, dtype = bool)
newpeak[1:] = aux[1:]-aux[:-1] > threshold
#locmax = locmax[newpeak]#easiest vsn, but not ideal - takes first point for each peak (ideally should average)

#better vsn:
ind = np.where(newpeak[:,0])[0]
locmax = np.empty(ind.shape[0]) 
for i in range(ind.shape[0]):
    if i == ind.shape[0]-1:
        locmax[i] = np.mean(aux[ind[i]:])
    else:
        locmax[i] = np.mean(aux[ind[i]:ind[i+1]])
      

# Hartigan dip test - test for multimodality, get intervals
bspeeds = bursts.burstHt.values
[counts,edges] = np.histogram(bspeeds, nbins, density = True)
xx = np.sort(bspeeds)
intervals = UniDip(xx, alpha=0.05, mrg_dst=.02).run()
#print(intervals)



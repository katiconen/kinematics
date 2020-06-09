# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:01:23 2020

@author: katic
"""
import pandas as pd
import os
import numpy as np
from sklearn.mixture import GaussianMixture as GM
from sklearn.neighbors import KernelDensity as KD
import matplotlib.pyplot as plt
from unidip import UniDip
import scipy.stats
import seaborn as sns
plt.rcParams.update({'font.size': 18})

pathname = r'C:\Users\katic\OneDrive\Documents\MPI\FishVR\data\mixed'
sesslistName  = pathname+'\\'+"sessionlist_LL.txt"
filesave = "popdata_selectedLL_3"

preVF = True
smoothdata = True
doPlots = True
mintime = 6000

listExists = os.path.isfile(sesslistName)
sessions = [] 
if listExists:
    sessions = [line.rstrip('\n') for line in open(sesslistName)]

#bursttist = pd.DataFrame(columns = ["GMM_mean1","GMM_mean2","GMM_variance1","GMM_variance2","KDE_firstpeak","KDE_lastpeak","KDE_npeaks","n_intervals_dip"], index = sessions)
bursttist = pd.DataFrame(columns = ["ave_burst_speed","ave_burst_time","time_speed_corr","corr_pval", "KDE_firstpeak","KDE_lastpeak","KDE_npeaks","n_intervals_dip"], index = sessions)
preVF_flag = np.zeros(len(sessions))

for session in sessions:
    
    #load burst data if available; otherwise generate from VR output csv
    try:
        bursts = pd.read_csv(pathname+'\\'+session+ '_burstmatEx.csv')
        
    except("Burst data for $session not found; generating from VR file"): 
        import get_kinematics as kin
        
        fishdata = pd.read_csv(pathname+'\\'+session+'.csv.gz')      
        if preVF:
            if np.any(fishdata.columns == "osg_fish1_y"):
                if np.all(fishdata.osg_fish1_y ==0):
                    preVF_flag = 1
                else:
                    ind = np.arange(0,np.where(fishdata.osg_fish1_y !=0)[0][0])
                    fishdata = fishdata.iloc[ind]
                    preVF_flag = 1
            else:
                preVF_flag = -1
       
        if fishdata.shape[0] < mintime:
            continue
        
        pos = np.empty([fishdata.shape[0],3])
        pos[:,0:3] = fishdata[['x','y','z']].to_numpy()
        
        time = np.empty(fishdata.shape[0])
        time[:] = fishdata.framenumber.to_numpy() - fishdata.loc[0,'framenumber0']
        time = time * 0.01
        ori = kin.getOri(pos)
        speed = kin.getSpeed(pos,time)
    
        newvals  = kin.cleanData(pos = pos[1:], ori = ori,\
                              speed = speed, time=time[1:],\
                              doSmooth = smoothdata, nbinsToAve = 5, interpSpdGaps = False)
        
        bursts,burstrate= kin.getBGcycle(speed,time,burstHtthresh = 0.0,threshold = 0.01)
        bursts.to_csv(pathname+'\\'+session+'_burstmatEx.csv')
        burstrate.to_csv(pathname+'\\'+session+'_burstrateEx.csv')
        
        speed = speed.reshape(-1,1)
        ori = ori.reshape(-1,1)
        time = time.reshape(-1,1)
        
        kindata = pd.DataFrame(np.hstack([pos[1:,:],speed,ori,time[1:]]),columns = ["X","Y","Z","Speed","Orientation","Time"])
        kindata.to_csv(pathname+'\\'+session+ "_kindata.csv")

    
##   evaluate burstheight distr - normal mixtures; gaussian kernel, unidip; maybe save fig(s)
#    #normal mixtures
#    bspeeds = bursts.burstHt.values.reshape(-1,1)
#    nparts = 2
#    model = GM(nparts).fit(bspeeds)
#    aic = model.aic(bspeeds)
#    bic = model.bic(bspeeds)
#    peakmeans = model.means_
#    peakvar = model.covariances_
#    
#    if doPlots:
#        fig = plt.figure(figsize=(8, 8))
#        x = np.linspace(0, .20, 1000)
#        logprob = model.score_samples(x.reshape(-1, 1))
#        responsibilities = model.predict_proba(x.reshape(-1, 1))
#        pdf = np.exp(logprob)
#        pdf_individual = responsibilities * pdf[:, np.newaxis]
#        plt.hist(bspeeds, 70, density=True, histtype='stepfilled', alpha=0.4)
#        plt.plot(x, pdf, '-k')
#        plt.plot(x, pdf_individual, '--k')
#        ax = plt.gca()
#        ax.set_xlabel('$speed$')
#        ax.set_ylabel('$p(speed)$')
        
    #kernel density estimates
    bspeeds = bursts.burstHt.values.reshape(-1,1)
    kdmodel = KD(bandwidth = .008, kernel = "gaussian")
    sample = bspeeds
    kdmodel.fit(sample)
    values = x
    values = values.reshape((len(values), 1))
    probabilities = kdmodel.score_samples(values)
    probabilities = np.exp(probabilities)

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

    if doPlots:
        fig = plt.figure(figsize=(8, 8))
        plt.hist(bspeeds, bins = 70, range = (0.,0.3), density = True, histtype='stepfilled', alpha=0.4)
        ax = plt.gca()
        ax.set_xlabel('$speed$')
        ax.set_ylabel('$p(speed)$')
        ax.plot(values[:], probabilities)
        ax.legend(["KDE"])
        f = plt.gcf()
        ax.set_xlim([-.01,0.3])
        f.savefig(pathname+'\\'+session + '_' + "burstHtDistr" + '.png', bbox_inches='tight')
        plt.close()
                   
    # Hartigan dip test - test for multimodality, get intervals
    bspeeds = bursts.burstHt.values
    xx = np.sort(bspeeds)
    intervals = UniDip(xx, alpha=0.05, mrg_dst=.02).run()
    nn = len(intervals)
   
    bursts["riseTime"] = np.round(bursts.peakTime - bursts.valleyTime,2)
    
    if doPlots:
        f = plt.figure(figsize=(8,8))
        ax = sns.lineplot(bursts.riseTime, bursts.burstHt)
        ax.set_ylim(0,0.17)
        ax.set_xlim(0,0.2)
        ax.set_yticks(np.arange(0,0.16,0.05))
        ax.set_xticks(np.arange(0,0.21,0.1))
        ax.set_xlabel('Burst Duration (s)')
        ax.set_ylabel('Peak Speed (m/s)')
        f.savefig(pathname+'\\'+session+ "burstTimeVsSpeed")
        plt.close()

    corrval, pval = scipy.stats.pearsonr(bursts.burstHt,bursts.riseTime)
    
    bursttist.loc[session] =  bursts.burstHt.mean(), bursts.riseTime.mean(), corrval, pval, locmax[0], locmax[-1], locmax.size, nn
    
bursttist["preVF_flag"] = preVF_flag
bursttist.dropna(inplace = True)
bursttist.to_csv(filesave + ".csv")

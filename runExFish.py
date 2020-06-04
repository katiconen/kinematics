# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:00:53 2020

@author: katic
"""

#import jnk
#
##jnk.main()
#
#x,_ = jnk.main(happiness = 1, variableface = "yours", blabhabl = "blah")
#
#print(x)

import get_kinematics as kin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

#
#fakex = np.arange(10.)
#pos = np.vstack((fakex,fakex,fakex)).T
#pos = kin.checkInputs(pos,xyOnly = True)
##print(pos)
##
#
##pos = np.vstack(([0.,0.],pos))
#ori = kin.getOri(pos)
##print(ori)
#
#speed = kin.getSpeed(pos)
##print(speed)
#
#pos = pos[1:,:]

#exec(open("makeFakeData.py").read())
doPlots = True

#open data csv
pathname = r'C:\Users\katic\OneDrive\Documents\MPI\FishVR\data\forDebug'
session = 'test_data0.csv.gz'
fishdata = pd.read_csv(pathname+'\\'+session)

ind_noVF = np.arange(0,np.where(fishdata.osg_fish1_y !=0)[0][0]) #if only pre-stimulus data is desired
fishdata = fishdata.iloc[ind_noVF]

## list column names
#for col in fishdata.columns:
#    print(col)
#    
#
##define pos, speed, ori, time, ideally as np arrays with a timepoint in each row
pos= np.empty([fishdata.shape[0],3])
pos[:,0:3] = fishdata[['x','y','z']].to_numpy()

time = np.empty(fishdata.shape[0])
time[:] = fishdata.framenumber.to_numpy() - fishdata.loc[0,'framenumber0']
time = time * 0.01

ori = kin.getOri(pos)
speed = kin.getSpeed(pos,time)

if doPlots:
    plt.close('all')
    f, axs = plt.subplots(3,2,figsize=(16,24))
    ax1,ax2,ax3,ax4,ax5,ax6 = axs.flatten()

    #histograms of speed and ori
    nbins = 500
    [counts,edges] = np.histogram(speed, nbins, density = True)
    ax1.hist(speed, edges, label='Speed')
    ax1.set_title('Speed')

    [counts,edges] = np.histogram(ori, nbins, density = True)
    ax6.hist(ori, edges, label='Ori')
    ax6.set_title('Ori')
    
    twinsize = 1000 #
    twinstart = 8500 #or pick an error flag, do that minus some lag to see how error looks
    ind = np.arange(twinstart,twinsize+twinstart)
    
    #plot speed vs time or counter
    ax2.plot(speed[ind])
    ax2.set_title('Speed')
    
    #plot pos xy trajectory in twin
    ax3.plot(pos[ind,0],pos[ind,1])
    ax3.set_title('XY Trajectory')
    
    #plot ori over time in twin (or vs counter)
    ax4.plot(ori[ind])
    ax4.set_title('Orientation')
    
    # plot time vs counter in twin
    ax5.plot(time[ind])
    ax5.set_title('Time')


newvals  = kin.cleanData(pos = pos[1:], ori = ori,\
                      speed = speed, time=time[1:],\
                      doSmooth = True, nbinsToAve = 5 )

bursts,burstrate= kin.getBGcycle(speed,time,burstHtthresh = 0.0,threshold = 0.01)

inact = kin.getSwimTime(BGmat = bursts.copy(), spThresh = 0.02, tThresh = 3.0) #speed = speed, time = time, 

oriAtBurst = kin.getTurns(ori,bursts,time) 

bursts.to_csv('burstmatEx.csv')
burstrate.to_csv('burstrateEx.csv')

#plot speed and ori
f0, axs = plt.subplots(3,1, figsize =(16,10))
ax0,ax1,ax2 = axs.flatten()

twinsize = 1000
twinstart = 0
ind = np.arange(twinstart,twinstart+twinsize)
ax0.plot(time[ind], speed[ind])
#plot burst valleys and peaks as vertical lines
bstart = bursts.loc[(bursts['valleyTime'] > time[twinstart]) & (bursts['valleyTime'] < time[twinstart + twinsize]),['valleyTime']].values
bpeak = bursts.loc[(bursts['valleyTime'] > time[twinstart]) & (bursts['valleyTime'] < time[twinstart + twinsize]),['peakTime']].values
bb = np.hstack([bstart,bpeak])
for xc in bb:
    ax0.axvline(x = xc[0],color='deepskyblue', alpha = 0.5)
    ax0.axvline(x = xc[1],color='mediumseagreen', alpha = 0.5)

ax1.plot(time[ind], ori[ind])
for xc in bb:
    ax1.axvline(x = xc[0],color='deepskyblue', alpha = 0.5)
    ax1.axvline(x = xc[1],color='mediumseagreen', alpha = 0.5)

nturns = 5
for i in range(nturns):
    ax2.plot(oriAtBurst[i,:])
    
plt.figure()
bursts.burstHt.hist(bins=100)
#plot first xx ori at burst

#kinMat, bursts, burstrate, whenInact, oriAlignedBurst, dataGaps = kin.runAll(pos)

# # for testing/debug
#f0, axs0 = plt.subplots(3,2, figsize=(16,24))
#ax11,ax21,ax31,ax41,ax51,ax61 = axs0.flatten()
#
#twinsize = 1000 #
#twinstart = 8500 #or pick an error flag, do that minus some lag to see how error looks
#ind = np.arange(twinstart,twinsize+twinstart)
#
#ax11.plot(x[ind,8-5])
#ax11.set_title('Ori Gaps')
#
##plot speed vs time or counter
#ax31.plot(ori[ind])
#ax31.set_title('Ori')
#
##plot pos xy trajectory in twin
#
#
##plot ori over time in twin (or vs counter)
#ax21.plot(x[ind,0])
#ax21.set_title('Pos Jumps')
#
##plot ori over time in twin (or vs counter)
#ax61.plot(pos[ind,1])
#ax61.set_title('Y')
#
## plot time vs counter in twin
#ax41.plot(pos[ind,0])
#ax41.set_title('X')
#
#ax51.plot(x[ind,9-5])
#ax51.set_title('Adj. Time')


# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:19:30 2020

@author: katic
"""
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()

doPlots = True

#speed
speedcycle = pd.read_csv('BG_onecycle.csv',header = None) #import BG speed
speedcycle = speedcycle.to_numpy()
speed = np.hstack((speedcycle,speedcycle,speedcycle,speedcycle))
speed = np.hstack((speed,speed,speed,speed,speed,speed,speed))
speed = np.hstack((speed,speed,speed,speed,speed,speed,speed))

#time
time = np.arange(0.,speed.size*0.01,0.01)

#ori: start random 0:2pi, change by random 0:0.02 radians
oristart = random.random()*2*np.pi
oristep = (np.random.random((1,speed.size))-0.5)*0.5
ori = np.cumsum(oristep) + oristart
ori = ori%(2*np.pi)-np.pi

#pos: start 0.0, update stepwise w speed and ori
pos = np.zeros((3,speed.size))
dx = np.multiply(speed,np.cos(ori))
dy = np.multiply(speed, np.sin(ori))

pos[0,:] = np.cumsum(dx,axis= 1)
pos[1,:] = np.cumsum(dy,axis= 1)
pos[2,:] = -0.03

#add noise
#random gaussian mean 0 std 0.001 for speed and pos, .02 for ori
noise = np.random.normal(scale = 0.001, size = (pos.shape))
pos = pos + noise
noise = np.random.normal(scale = 0.001, size = (speed.size))
speed = speed + noise
noise = np.random.normal(scale = 0.01, size = (ori.size))
ori = ori + noise


#add artifact missed steps w overshoot in speed
aux = np.random.random((speed.size))
aux[0] = 1.0
aux[-1] = 1.0
ind_m = np.where(aux < 0.05)[0] #~5% of data points
pos[:,ind_m] = pos[:,ind_m-1]
ori[ind_m] = ori[ind_m-1]
speed[:,ind_m] = 0.
speed[:,ind_m+1] = 2.0*speed[:,ind_m+1]



#add breaks in trajectory - short
#pick random points and random (short) length, set vals to arbitrary 
aux = np.random.random((speed.size))
aux[0] = 1.0
aux[-3:] = 1.0
ind_sb = np.where(aux < 0.01)[0] #~1% of data points
ind_sb = np.vstack((ind_sb,ind_sb+1,ind_sb+2,ind_sb+3))
pos[:,ind_sb] = pos[:,ind_sb]*0.1
ori[ind_sb] = ori[ind_sb]*0.1
speed[:,ind_sb] = speed[:,ind_sb]*0.5
time[ind_sb] = 0.0


#add breaks in trajectory - long
#easiest is just duplicate whole trajectory from start a couple times (or append part to end)
#alt: pick random pts to insert bits of trajectory; if this, remember to updtate ind_m and ind_sb
initsize = speed.size
pos = np.hstack((pos,pos,pos))
speed = np.hstack((speed,speed,speed))
ori = np.hstack((ori,ori,ori))
time = np.hstack((time,time*2.0,time*3.0))

#plots for random checks


if doPlots:
    #plot checks
    
    plt.close('all')

    f, axs = plt.subplots(3,2,figsize=(16,24))
    ax1,ax2,ax3,ax4,ax5,ax6 = axs.flatten()

    #histograms of speed and ori
    nbins = 500
    [counts,edges] = np.histogram(speed[0], nbins, density = True)
    ax1.hist(speed[0], edges, label='Fake Speed')
    ax1.set_title('Fake Speed')

    [counts,edges] = np.histogram(ori, nbins, density = True)
    ax6.hist(ori, edges, label='Fake Ori')
    ax6.set_title('Fake Ori')
    
    twinsize = 1000 #
    twinstart = 0 #or pick an error flag, do that minus some lag to see how error looks
    ind = np.arange(twinstart,twinsize)
    
    #plot speed vs time or counter
    ax2.plot(speed[0,ind])
    ax2.set_title('Fake Speed')
    
    #plot pos xy trajectory in twin
    ax3.plot(pos[0,ind],pos[1,ind])
    ax3.set_title('XY Trajectory')
    
    #plot ori over time in twin (or vs counter)
    ax4.plot(ori[ind])
    ax4.set_title('Orientation')
    
    # plot time vs counter in twin
    ax5.plot(time[ind])
    ax5.set_title('Time (with errors)')
    
        # plot dpos vs speed
#    dPos = np.subtract(pos[:,1:],pos[:,:-1])
#    dPos = np.linalg.norm(dPos, axis = 0)
#    dpos2 = np.sqrt(np.multiply(dx,dx)+np.multiply(dy,dy))
#    ax6.scatter(speed[:,1:],dPos, s = 1)

#save as csv?
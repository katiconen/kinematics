# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:10:28 2020

Goes to loopbio vr web control server and downloads VR data files based on experiment title (or substring)
Saves under usable names based on experiment name in VR (KC_onto_fishID_dpf_EXP_iter, e.g. KC_onto_0000_07_OM_it00)
Other script (sort_sessions.py) will sort into folders by fishID and age
Currently set to avoid re-saving files if they are alread on the session list
Possible to do's: 
        - confirm that new sessions can be appended to session list (and new data saved)
        - add check and patch/error for repeat experiment name
        - provide an option to delete and re-download all data (denovo) if desired
        - set to avoid re-downloading files even if they are not listed in the sessionlist (e.g. if list is deleted)
        - add info about vr-machine, date/time, etc to sessionlist (or other file)
                          
@author: katic
"""

#import libraries

import urllib.request, urllib.parse #, urllib.error
import os
from bs4 import BeautifulSoup
from pathlib import Path

#change to savedirectory; if directory does not exist, create it
savedir = 'C:/users/katic/onedrive/documents/mpi/fishvr/data/onto'
Path(savedir).mkdir(parents=True, exist_ok=True)
os.chdir(savedir)

#fetch data - get code from fetch_data.py
#modifications: save with readable name, filter based on whether there's a processed data file

experimentName = 'KC_test'# 'KC_onto'
useWholeName = False
subInd = len(experimentName)
dateRange = [20200101, 20200303] #yr month day

sesslistName = "sessionlist.txt"

listExists = os.path.isfile(sesslistName)
sessions = [] 
if listExists:
    sessions = [line.rstrip('\n') for line in open(sesslistName)]

urlBase = 'http://10.126.18.10:4280' #url for fishvr site; need to be on mpio_kn network to work
url = urlBase
dummy = 0

while 1:  #go through pages until last page reached - maybe change to stop early if !denovo - e.g. if already known file is encountered
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')
    trows = soup('tr')
    for trow in trows[1:]:      #ignore header row
        date = trow('td')[3].contents[0][:10]
        date = int(date.replace('-',''))
        if date < dateRange[0]:
            break
        elif date > dateRange[1]:
            continue
        uuid = trow('td')[1].contents[0]
        if not trow('td')[2].contents: #catch cases where no experiment title was assigned
            print('Warning: Experiment ' + uuid + ' has no Title')
            continue
        expTitle = trow('td')[2].contents[0] #title of experiment
        if not useWholeName:
            checkTitle = expTitle[:subInd]
        else:
            checkTitle = expTitle
        if checkTitle != experimentName:
            continue
        else:
            print(expTitle)
            if uuid in sessions:
                print('All new sessions added.\nStopping point: '+uuid)
                dummy = 1
                break #session already downloaded & added to list; 
            else:
                               
                #save datafile; to do: set to check for prev file by same name and skip (if !denovo) or replace (if denovo)
                try:
                    datafile = urllib.request.urlopen(urlBase+'/static_analysis/'+uuid+'/default.combine.csv.gz').read()
                except:
                    print('Processed data file does not exist for ' + uuid)
                    continue
                fhand = open(expTitle+'.csv.gz', 'wb')
                fhand.write(datafile)       
                fhand.close()
#                
#                dummy = 1
#                break
                
                sessions.append(expTitle) #add experiment to sessionlist
                
                html = urllib.request.urlopen(url).read() #return to previous page
                soup = BeautifulSoup(html, 'html.parser')
                #download data to relevant directory
                
    if dummy == 1:
        break
    
    # go to next page and repeat
    tags = soup('a')
    for tag in tags:
        if tag.get('aria-label') == 'Next':
            urlNext = tag.get('href')
            
    if urlBase+urlNext == url:
        break
    else:
        url = urlBase+urlNext


#save sessionlist
fhand = open("sessionlist.txt", 'w')
for sess in sessions:
    fhand.write(str(sess) + '\n')
fhand.close()


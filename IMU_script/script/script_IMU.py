#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 04:47:02 2021

@author: zackinho91
"""

import os
import pathlib
import matplotlib
import pandas as pd
import numpy as np
import scipy
from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.signal import find_peaks
import pickle
import tools
from tools import filter_data , lag_finder

def IMU(path,filename):
    """Return simple parameters on the IMU signal. The path of the IMU file must be given as input
    - path is the path of the file
    -filename is the name of the file
    
    calculated parameters : 
        - norms acceleration and angular velocity
        -signal speed threshold
        -peak of the signal
        -start and end time of the different activities
        sample_rate (int): Signal sampling rate calculated from the dataset
   
    Returns :
        dict: dictionary containing the segmented and paired start and end data of the activities.
    """
    df=pd.read_csv(path+'/'+filename)
    
    print(df.isnull().mean()*100)
    length=len(df)
    begin=df['timestamp_relative_[ms]'][0]
    end=df['timestamp_relative_[ms]'][length-1]
    fs=round(length/((end-begin)/1000),2)
    print('We have '+str(length)+' of observations for a period of '+str(round((end-begin)/1000,2))+ ' secondes. We therefore have an sample rate of ' + str(fs)+' Hz')
    
    print('Visualisation of data')
    plt.plot(df['acc_x_[m/s^2]'],'r',label='acc_x')
    plt.plot(df['acc_y_[m/s^2]'],'b',label='acc_y')
    plt.plot(df['acc_z_[m/s^2]'],'g',label='acc_z')
    plt.title('acceleration data x,y,z in m/s^2')
    plt.ylabel("acceleration in in m/s^2 ")
    plt.legend()
    plt.show()
    
    print('visualisation of filtred data')
    a=filter_data(df['acc_x_[m/s^2]'], 100, low_pass = 2, order = 4)
    plt.subplot(121)
    plt.plot(df['acc_x_[m/s^2]'])
    plt.title('raw acceleration in x axis ')
    plt.subplot(122)
    plt.plot(a)
    plt.title('filtred acceleration in x axis ')
    plt.show()
    
    
    df1=df[['timestamp_relative_[ms]','acc_x_[m/s^2]','acc_y_[m/s^2]','acc_z_[m/s^2]']]
    df2=df[['timestamp_relative_[ms]','gyr_x_[rad/s]', 'gyr_y_[rad/s]', 'gyr_z_[rad/s]']]
    
    print('Calculation of norms')

    acc=[]
    gyr=[]
    for i in range(len(df)): 
        arr = np.array([df1['acc_x_[m/s^2]'][i], df1['acc_y_[m/s^2]'][i],df1['acc_z_[m/s^2]'][i]])
        acc.append(norm(arr))
        arr1 = np.array([df2['gyr_x_[rad/s]'][i], df2['gyr_y_[rad/s]'][i],df2['gyr_z_[rad/s]'][i]])
        gyr.append(norm(arr1))
    data={'acc':acc,'gyr':gyr}
    data=pd.DataFrame(data)
    
    acc_filt=filter_data(data.acc, 200, low_pass = 2, order = 4)
    gyr_filt=filter_data(data.gyr, 200, low_pass = 2, order = 4)
    data['acc_filt']=acc_filt
    data['gyr_filt']=gyr_filt
    data['time']=df['timestamp_relative_[ms]']
    
    print('Plot of acceleration and angular activity filtred')
    plt.subplot(221)
    sns.lineplot(x=df1['timestamp_relative_[ms]'],y=data.acc)
    plt.title('acc')
    plt.subplot(222)
    sns.lineplot(x=df1['timestamp_relative_[ms]'],y=data.acc_filt,color='red')
    plt.title('acc_filt')
    plt.subplot(223)
    sns.lineplot(x=df1['timestamp_relative_[ms]'],y=data.gyr)
    plt.title('gyr_raw')
    plt.subplot(224)
    sns.lineplot(x=df1['timestamp_relative_[ms]'],y=data.gyr_filt,color='red')
    plt.title('gyr_filt')
    plt.show()
    
    #print('auto correlation of the accelerometer and gyrocope data')
    #lag=lag_finder(acc,gyr,fs)
    #if lag >-1 or lag<1:
    #    print('A maximum correlation is observed in O')
   # else:
    #    print('there is a lag')
        
    print('Observation of activities with acceleration')
    f, Pxx_spec = signal.welch(df1['acc_x_[m/s^2]'],fs,'flattop', nperseg=1024,scaling='spectrum')
    plt.figure()
    plt.plot(f,Pxx_spec)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Acceleration in x m/s^2')
    plt.xscale('log')
    plt.show()
    
    
    print("Detection peaks")
    from scipy.signal import find_peaks
    peaks, height = find_peaks(data.gyr_filt, height=2.9)
    np.diff(peaks)
    plt.plot(gyr_filt)
    plt.plot(peaks, data.gyr_filt[peaks], "x")
    plt.title('peaks of norm angular velocity filtred rad/s')

    plt.show()
    
    threshold=0.05*np.mean(height['peak_heights'])
    threshold
    activity=data[data.gyr_filt>threshold]
    
    print('peaks with acceleration')
    
    peaks, height= find_peaks(acc_filt, height=11)
    np.diff( peaks)
    plt.plot(acc_filt)
    plt.plot( peaks, acc_filt[ peaks], "x")
    plt.title('peaks of norm accelerometer filtred  m/s^2')
    plt.show()
    
    print('Determination of activities')
    z=[]
    for i,j in enumerate(peaks):
        if peaks[i]-peaks[i-1]<=5000:
            pass
        else:
            z.append([peaks[i-1],peaks[i]])
    
    cutter = {}
    for i in range(len(z)):
        if i == 0:
            cutter['activity'+str(i+1)] = [peaks[0],z[0][0]]
    
        else:
            cutter['activity'+str(i+1)] = [z[i-1][1],z[i][0]]        
    cutter['activity'+str(len(z)+1)] = [z[len(z)-1][1],peaks[len(peaks)-1]]
    print(cutter)
    plt.subplot(211)
    plt.plot(activity.gyr_filt,'r')
    plt.title('Sample segmented thnak to threshold of angular velocity [rad/s] ')
    plt.ylabel('velocity [rad/s]')
    plt.grid('on')
    plt.show()
    print('Time of activities')
    activities={}
    timer={}
    for i in range(len(cutter)):
        timer['start_'+str(i+1)] = data.time[cutter[list(cutter.keys())[i]][0]]
        timer['stop_'+str(i+1)] = data.time[cutter[list(cutter.keys())[i]][1]]
        print(list(cutter.keys())[i]+' starts at '+str(round(timer[list(timer.keys())[i]]/1000,2))+ ' seconds and finishes at '+ str(round(timer[list(timer.keys())[i+1]]/1000,2))+' seconds.')

    activities['timer']=timer
    print('plot activities)')
    for i in range(len(cutter)):
        if i==0 or i==1:
            plt.subplot(2,1,1)
            plt.plot(data.gyr_filt,'r')
            plt.title('norm of angular velocity of all the sample in red and norm of angular velocity of '+str(list(cutter.keys())[i])+' in blue')
            plt.subplot(2,1,2)
            plt.plot(data.gyr_filt[cutter[list(cutter.keys())[i]][0]:cutter[list(cutter.keys())[i]][1]],'c')
            plt.show()
        else:
            plt.subplot(2,1,1)
            plt.plot(data.gyr_filt,'r')
            plt.title('norm of angular velocity of all the raw data in red and norm of angular velocity of '+str(list(activities.keys())[i])+' sample in blue')
            plt.subplot(2,1,2)
            plt.plot(data.gyr_filt[cutter[list(cutter.keys())[i]][0]:cutter[list(cutter.keys())[i]][1]],'c')
            plt.show()
            
        print('Storage of the segmented data')
    
        for i in range(len(cutter)):
            activities[list(cutter.keys())[i]]=data[cutter[list(cutter.keys())[i]][0]:cutter[list(cutter.keys())[i]][1]]
        activities.keys()
        
        data_cleaned=os.makedirs(path+'/Data_cleaned')
        activities_file = open(path+'/Data_cleaned/activities.pkl', "wb")
        pickle.dump(activities, activities_file)
        activities_file.close()
        
    return timer


        
    



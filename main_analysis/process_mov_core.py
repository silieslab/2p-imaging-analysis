#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:42:35 2020

@authors: Burak Gur, Deniz Yuzak
"""


import copy
import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import pickle
import random
import warnings

from skimage import io
from scipy import interpolate
from scipy.stats.stats import pearsonr
from itertools import permutations
from roipoly import RoiPoly, MultiRoi
from skimage import filters
from scipy.ndimage.interpolation import rotate

import main_analysis.ROI_mod
#import summary_figures as sf
#from xmlUtilities import getFramePeriod, getLayerPosition, getPixelSize,getMicRelativeTime
#from core_functions import readStimOut, readStimInformation, getEpochCount, divide_all_epochs
#from core_functions import divideEpochs
#from post_analysis_core import run_matplotlib_params

def pre_processing_movie (dataDir,stimInputDir):
    

    # Generate necessary directories for figures
    current_t_series=os.path.basename(dataDir)
    
    # Load movie, get stimulus and imaging information
    try:
        movie_path = os.path.join(dataDir, 'motCorr.sima',
                                  '{t_name}_motCorr.tif'.format(t_name=current_t_series))
        time_series = io.imread(movie_path)
    except IOError:
        movie_path = os.path.join(dataDir, '{t_name}_C2_time_series.tif'.format(t_name=current_t_series))
        time_series = io.imread(movie_path)
        
    ## Get stimulus and xml information
    (stimulus_information, imaging_information) = \
        get_stim_xml_params(dataDir, stimInputDir)
    
    
    
    return time_series, stimulus_information,imaging_information

def pre_processing_moviePyStim(dataDir,stimInputDir):
    

    # Generate necessary directories for figures
    current_t_series=os.path.basename(dataDir)
    
    # Load movie, get stimulus and imaging information
    try:
        movie_path = os.path.join(dataDir, 'motCorr.sima',
                                  '{t_name}_motCorr.tif'.format(t_name=current_t_series))
        time_series = io.imread(movie_path)
    except IOError:
        movie_path = os.path.join(dataDir, '{t_name}_C2_time_series.tif'.format(t_name=current_t_series))
        time_series = io.imread(movie_path)
        
    ## Get stimulus and xml information
    (stimulus_information, imaging_information) = \
        get_stim_xml_paramsPyStim(dataDir, stimInputDir)
    
    
    
    return time_series, stimulus_information,imaging_information
    
def compute_correlation_image(video):
    
    xdim = video.shape[1]
    ydim = video.shape[2]
    window = 6
    factor = window/2
    corr_image = np.zeros(video.shape[1:])
    pval_image = np.zeros(video.shape[1:])
    for ix in range(xdim- window):
        for iy in range(ydim-window):
            pix_x = ix+factor
            pix_y = iy+factor
            curr_pix_trace = video[:,pix_x,pix_y]
            neighbors = video[:,pix_x-factor:pix_x+factor, 
                              pix_y-factor:pix_y+factor]
            neighbors_trace = neighbors.mean(axis=1).mean(axis=1)
            curr_coeff, pval = pearsonr(curr_pix_trace,neighbors_trace)
            corr_image[pix_x,pix_y] = curr_coeff
            pval_image[pix_x,pix_y] = pval
    return corr_image, pval_image
    

def get_stim_xml_params(t_series_path, stimInputDir):
    """ Gets the required stimulus and imaging parameters.
    Parameters
    ==========
    t_series_path : str
        Path to the T series folder for retrieving stimulus related information and
        xml file which contains imaging parameters.
    
    stimInputDir : str
        Path to the folder where stimulus input information is located.
        
    Returns
    =======
    stimulus_information : list 
        Stimulus related information is stored here.
    trialCoor : list
        Start, end coordinates for each trial for each epoch
    frameRate : float
        Image acquisiton rate.
    depth :
        Z axis value of the imaging dataset.

    """

    # Finding the xml file and retrieving relevant information
    
    xmlPath = os.path.join(t_series_path, '*-???.xml')
    xmlFile = (glob.glob(xmlPath))[0]
    
    #  Finding the frame period (1/FPS) and layer position
    framePeriod = getFramePeriod(xmlFile=xmlFile)
    frameRate = 1/framePeriod
    layerPosition = getLayerPosition(xmlFile=xmlFile)
    depth = layerPosition[2]
    
    imagetimes = getMicRelativeTime(xmlFile)
    
    # Pixel definitions
    x_size, y_size, pixelArea = getPixelSize(xmlFile)
    
    # Stimulus output information
    
    stimOutPath = os.path.join(t_series_path, '_stimulus_output_*')
    stimOutFile = (glob.glob(stimOutPath))[0]
    (stimType, rawStimData) = readStimOut(stimOutFile=stimOutFile, 
                                          skipHeader=1)
    
    # Stimulus information
    (stimInputFile,stimInputData) = readStimInformation(stimType=stimType,
                                                      stimInputDir=stimInputDir)
    isRandom = int(stimInputData['Stimulus.randomize'][0])
    epochDur = stimInputData['Stimulus.duration']
    epochDur = [float(sec) for sec in epochDur]
    epochCount = getEpochCount(rawStimData=rawStimData, epochColumn=3)
    # Finding epoch coordinates and number of trials, if isRandom is 1 then
    # there is a baseline epoch otherwise there is no baseline epoch even 
    # if isRandom = 2 (which randomizes all epochs)                                        
    if epochCount <= 1:
        trialCoor = 0
        trialCount = 0
    elif isRandom == 1:
        (trialCoor, trialCount, _) = divideEpochs(rawStimData=rawStimData,
                                                 epochCount=epochCount,
                                                 isRandom=isRandom,
                                                 framePeriod=framePeriod,
                                                 trialDiff=0.20,
                                                 overlappingFrames=0,
                                                 firstEpochIdx=0,
                                                 epochColumn=3,
                                                 imgFrameColumn=7,
                                                 incNextEpoch=True,
                                                 checkLastTrialLen=True)
    else:
        (trialCoor, trialCount) = divide_all_epochs(rawStimData, epochCount, 
                                                    framePeriod, trialDiff=0.20,
                                                    epochColumn=3, imgFrameColumn=7,
                                                    checkLastTrialLen=True)
     
    stimulus_information ={}
    stimulus_data = stimInputData
    stimulus_information['epoch_dur'] = epochDur
    stimulus_information['random'] = isRandom
    stimulus_information['output_data'] = rawStimData
    stimulus_information['frame_timings'] = imagetimes
    if isRandom==0:
        stimulus_information['baseline_epoch'] = 0  
        stimulus_information['baseline_duration'] = \
            stimulus_information['epoch_dur'][stimulus_information['baseline_epoch']]
        stimulus_information['epoch_adjuster'] = 0
        print('\n Stimulus non random, baseline epoch selected as 0th epoch\n')
    elif isRandom == 2:
        stimulus_information['baseline_epoch'] = None
        stimulus_information['baseline_duration'] = None
        stimulus_information['epoch_adjuster'] = 0
        print('\n Stimulus all random, no baseline epoch present\n')
    elif isRandom == 1:
        stimulus_information['baseline_epoch'] = 0 
        stimulus_information['baseline_duration'] = \
            stimulus_information['epoch_dur'][stimulus_information['baseline_epoch']]
        stimulus_information['epoch_adjuster'] = 1
        
    stimulus_information['epoch_dir'] = \
            np.asfarray(stimulus_data['Stimulus.stimrot.mean'])
    epoch_speeds = np.asfarray(stimulus_data['Stimulus.stimtrans.mean'])
    stimulus_information['epoch_frequency'] = \
        epoch_speeds/np.asfarray(stimulus_data['Stimulus.spacing'])
    stimulus_information['epochs_duration'] =\
         np.asfarray(stimulus_data['Stimulus.duration'])
    stimulus_information['epoch_number'] =  \
        np.asfarray(stimulus_data['EPOCHS'][0])
    stimulus_information['stim_type'] =  \
        np.asfarray(stimulus_data['Stimulus.stimtype'])
    stimulus_information['input_data'] = stimInputData
    stimulus_information['stim_name'] = stimType.split('\\')[-1]
    stimulus_information['trial_coordinates'] = trialCoor
    
    imaging_information = {'frame_rate' : frameRate, 'pixel_size': x_size, 
                             'depth' : depth}
        
    return stimulus_information, imaging_information

def get_stim_xml_paramsPyStim(t_series_path, stimInputDir):
    """ Gets the required stimulus and imaging parameters.
    Parameters
    ==========
    t_series_path : str
        Path to the T series folder for retrieving stimulus related information and
        xml file which contains imaging parameters.
    
    stimInputDir : str
        Path to the folder where stimulus input information is located.
        
    Returns
    =======
    stimulus_information : list 
        Stimulus related information is stored here.
    trialCoor : list
        Start, end coordinates for each trial for each epoch
    frameRate : float
        Image acquisiton rate.
    depth :
        Z axis value of the imaging dataset.

    """

    # Finding the xml file and retrieving relevant information
    
    xmlPath = os.path.join(t_series_path, '*-???.xml')
    xmlFile = (glob.glob(xmlPath))[0]
    
    #  Finding the frame period (1/FPS) and layer position
    framePeriod = getFramePeriod(xmlFile=xmlFile)
    frameRate = 1/framePeriod
    layerPosition = getLayerPosition(xmlFile=xmlFile)
    depth = layerPosition[2]
    
    imagetimes = getMicRelativeTime(xmlFile)
    
    # Pixel definitions
    x_size, y_size, pixelArea = getPixelSize(xmlFile)
    
    # Stimulus information
    stimOutPath = os.path.join(t_series_path, '*.pickle')
    stimOutFile = (glob.glob(stimOutPath))[0]

    load_path = open(stimOutFile, 'rb')
    stimInfo = pickle.load(load_path)

    randomization_condition = int(stimInfo['meta']['randomization_condition'])
    epochDur = stimInputData['Stimulus.duration']
    epochDur = [float(sec) for sec in epochDur]
    epochCount = getEpochCount(rawStimData=rawStimData, epochColumn=3)
    # Finding epoch coordinates and number of trials, if isRandom is 1 then
    # there is a baseline epoch otherwise there is no baseline epoch even 
    # if isRandom = 2 (which randomizes all epochs)                                        
    if epochCount <= 1:
        trialCoor = 0
        trialCount = 0
    elif randomization_condition == 1:
        (trialCoor, trialCount, _) = divideEpochs(rawStimData=rawStimData,
                                                 epochCount=epochCount,
                                                 isRandom=randomization_condition,
                                                 framePeriod=framePeriod,
                                                 trialDiff=0.20,
                                                 overlappingFrames=0,
                                                 firstEpochIdx=0,
                                                 epochColumn=3,
                                                 imgFrameColumn=7,
                                                 incNextEpoch=True,
                                                 checkLastTrialLen=True)
    else:
        (trialCoor, trialCount) = divide_all_epochs(rawStimData, epochCount, 
                                                    framePeriod, trialDiff=0.20,
                                                    epochColumn=3, imgFrameColumn=7,
                                                    checkLastTrialLen=True)
     
    stimulus_information ={}
    stimulus_data = stimInputData
    stimulus_information['epoch_dur'] = epochDur
    stimulus_information['random'] = randomization_condition
    stimulus_information['output_data'] = rawStimData
    stimulus_information['frame_timings'] = imagetimes
    if randomization_condition==0:
        stimulus_information['baseline_epoch'] = 0  
        stimulus_information['baseline_duration'] = \
            stimulus_information['epoch_dur'][stimulus_information['baseline_epoch']]
        stimulus_information['epoch_adjuster'] = 0
        print('\n Stimulus non random, baseline epoch selected as 0th epoch\n')
    elif randomization_condition == 2:
        stimulus_information['baseline_epoch'] = None
        stimulus_information['baseline_duration'] = None
        stimulus_information['epoch_adjuster'] = 0
        print('\n Stimulus all random, no baseline epoch present\n')
    elif randomization_condition == 1:
        stimulus_information['baseline_epoch'] = 0 
        stimulus_information['baseline_duration'] = \
            stimulus_information['epoch_dur'][stimulus_information['baseline_epoch']]
        stimulus_information['epoch_adjuster'] = 1
        
    stimulus_information['epoch_dir'] = \
            np.asfarray(stimulus_data['Stimulus.stimrot.mean'])
    epoch_speeds = np.asfarray(stimulus_data['Stimulus.stimtrans.mean'])
    stimulus_information['epoch_frequency'] = \
        epoch_speeds/np.asfarray(stimulus_data['Stimulus.spacing'])
    stimulus_information['epochs_duration'] =\
         np.asfarray(stimulus_data['Stimulus.duration'])
    stimulus_information['epoch_number'] =  \
        np.asfarray(stimulus_data['EPOCHS'][0])
    stimulus_information['stim_type'] =  \
        np.asfarray(stimulus_data['Stimulus.stimtype'])
    stimulus_information['input_data'] = stimInputData
    stimulus_information['stim_name'] = stimType.split('\\')[-1]
    stimulus_information['trial_coordinates'] = trialCoor
    
    imaging_information = {'frame_rate' : frameRate, 'pixel_size': x_size, 
                             'depth' : depth}
        
    return stimulus_information, imaging_information

def separate_trials_video(time_series,stimulus_information,frameRate):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces.
    
    Parameters
    ==========
    time_series : numpy array
        Time series in the form of: frames x m x n (m & n are pixel dimensions)
    
    trialCoor : list
        Start, end coordinates for each trial for each epoch
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    dff_baseline_dur_frame: int
        Duration of baseline before the stimulus for using in dF/F calculation.
        
    Returns
    =======
    wholeTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            baseline epoch - stimulus epoch - baseline epoch
    respTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-

    """
    
    trialCoor = stimulus_information['trial_coordinates']
    mov_xDim = time_series.shape[1]
    mov_yDim = time_series.shape[2]
    wholeTraces_allTrials = {}
    respTraces_allTrials = {}
    baselineTraces_allTrials = {}
    for iEpoch in trialCoor:
        currentEpoch = trialCoor[iEpoch]
        current_epoch_dur = stimulus_information['epochs_duration'][iEpoch]
        trial_numbers = len(currentEpoch)
        trial_lens = []
        base_lens = []
        for curr_trial_coor in currentEpoch:
            current_trial_length = curr_trial_coor[0][1]-curr_trial_coor[0][0]
            trial_lens.append(current_trial_length)
            
            baselineStart = curr_trial_coor[1][0]
            baselineEnd = curr_trial_coor[1][1]
            
            base_lens.append(baselineEnd - baselineStart) 
     
            
        trial_len =  min(trial_lens)-4
        resp_len = int(round(frameRate * current_epoch_dur))+1
#        resp_len = int(round(frameRate * 3))
        base_len = min(base_lens)
        
        if stimulus_information['random']==2:
            wholeTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                               trial_numbers))
            respTraces_allTrials[iEpoch] = None
            baselineTraces_allTrials[iEpoch] = None
            
        elif stimulus_information['random'] == 1:
            wholeTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                               trial_numbers))
            respTraces_allTrials[iEpoch] = np.zeros(shape=(resp_len,mov_xDim,mov_yDim,
                               trial_numbers))
            baselineTraces_allTrials[iEpoch] = np.zeros(shape=(base_len,mov_xDim,mov_yDim,
                               trial_numbers))
        else:
            wholeTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                               trial_numbers))
            respTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                               trial_numbers))
            base_len  = np.shape(wholeTraces_allTrials\
                                 [stimulus_information['baseline_epoch']])[0]
            baselineTraces_allTrials[iEpoch] = np.zeros(shape=(base_len,mov_xDim,mov_yDim,
                               trial_numbers))
                
        
        for trial_num , current_trial_coor in enumerate(currentEpoch):
            
            if stimulus_information['random']==2:
                trialStart = current_trial_coor[0][0]
                trialEnd = current_trial_coor[0][1]
                raw_signal = time_series[trialStart:trialEnd, : , :]
                
                wholeTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
               
            elif stimulus_information['random'] == 1:
                trialStart = current_trial_coor[0][0]
                trialEnd = current_trial_coor[0][1]
                
                baselineStart = current_trial_coor[1][0]
                baselineEnd = current_trial_coor[1][1]
                
                respStart = current_trial_coor[1][1]
                epochEnd = current_trial_coor[0][1]
                
                raw_signal = time_series[trialStart:trialEnd, : , :]
            
                currentResp = time_series[respStart:epochEnd, : , :]
                   #        dffTraces_allTrials[iEpoch].append(dFF[:trial_len,:,:])
                wholeTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
                respTraces_allTrials[iEpoch][:,:,:,trial_num]= currentResp[:resp_len,:,:]
                baselineTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:base_len,:,:]
            else:
                
                # If the sequence is non random  the trials are just separated without any baseline
                trialStart = current_trial_coor[0][0]
                trialEnd = current_trial_coor[0][1]
                if iEpoch == stimulus_information['baseline_epoch']:
                    baseline_signal = time_series[trialStart:trialEnd, : , :]
                raw_signal = time_series[trialStart:trialEnd, : , :]
                
                wholeTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
                respTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
                baselineTraces_allTrials[iEpoch][:,:,:,trial_num]= baseline_signal[:base_len,:,:]
            
           
            
            
 
        
        print('Epoch %d completed \n' % iEpoch)
        
    return (wholeTraces_allTrials, respTraces_allTrials, baselineTraces_allTrials)

def calculate_pixel_SNR(baselineTraces_allTrials,respTraces_allTrials,
                  stimulus_information,frameRate,SNR_mode ='Estimate'):
    """ Calculates the pixel-wise signal-to-noise ratio (SNR). Equation taken from
    Kouvalainen et al. 1994 (see calculation of SNR true from SNR estimated). 
    
    Parameters
    ==========
    respTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-
            
    stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    SNR_mode : not implemented yet
    
        
        
    Returns
    =======
    
    SNR_max_matrix : np array
        An m x n array with pixel-wise SNR.

    """
    
    mov_xDim = np.shape(baselineTraces_allTrials[1])[1]
    mov_yDim = np.shape(baselineTraces_allTrials[1])[2]
    total_epoch_numbers = len(baselineTraces_allTrials)
    
    
#    total_background_dur = stimulus_information['epochs_duration'][0]
    SNR_matrix = np.zeros(shape=(mov_xDim,mov_yDim,total_epoch_numbers))
    for iPlot, iEpoch in enumerate(baselineTraces_allTrials):
        
        trial_numbers = np.shape(baselineTraces_allTrials[iEpoch])[3]
        currentBaseTrace = baselineTraces_allTrials[iEpoch][:,:,:,:]
        currentRespTrace =  respTraces_allTrials[iEpoch][:,:,:,:]
        
        noise_std = currentBaseTrace.std(axis=0).mean(axis=2)
        resp_std = currentRespTrace.std(axis=0).mean(axis=2)
        
        signal_std = resp_std - noise_std
        # SNR calculation taken from
        curr_SNR_true = ((trial_numbers+1)/trial_numbers)*(signal_std/noise_std) - 1/trial_numbers
#        curr_SNR = (signal_std/noise_std) 
        SNR_matrix[:,:,iPlot] = curr_SNR_true
        
       
    SNR_matrix[np.isnan(SNR_matrix)] = np.nanmin(SNR_matrix) # change nan values with min values
    
    SNR_max_matrix = SNR_matrix.max(axis=2) # Take max SNR for every pixel for every epoch

    return SNR_max_matrix

def calculate_pixel_max(respTraces_allTrials,stimulus_information):
    
    """ Calculates the pixel-wise maximum responses for each epoch. Returns max 
    epoch indices as well but adjusting the indices considering that baseline is 0.
    
    Parameters
    ==========
    respTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
        
    Returns
    =======
    
    MaxResp_matrix_without_edge : np array
        An array with pixel-wise maxiumum responses for epochs other than edges
        that are normally used as probe stimuli. Edge maximums are set to -100 so that
        they're never maximum.
    
    MaxResp_matrix_all_epochs : np array
        An array with pixel-wise maxiumum responses for every epoch.
        
    maxEpochIdx_matrix_without_edge : np array
        An array with pixel-wise maximum epoch indices. Adjusts the indices considering
        that baseline index is 0 and epochs start from 1.
        
    maxEpochIdx_matrix_all : np array
        An array with pixel-wise maximum epoch indices. Adjusts the indices considering
        that baseline index is 0 and epochs start from 1.
        
    """
    epoch_adjuster = stimulus_information['epoch_adjuster']
    
    mov_xDim = np.shape(respTraces_allTrials[1])[1]
    mov_yDim = np.shape(respTraces_allTrials[1])[2]
    total_epoch_numbers = len(respTraces_allTrials)
    
    # Create an epoch-wise maximum response list
    maxResp = {}
    meanResp ={}
    # Create an array with m x n x nEpochs
    MaxResp_matrix = np.zeros(shape=(mov_xDim,mov_yDim,total_epoch_numbers))
    MeanResp_matrix = np.zeros(shape=(mov_xDim,mov_yDim,total_epoch_numbers))
    
    for index, iEpoch in enumerate(respTraces_allTrials):
        
        # Find maximum of pixels after trial averaging
        curr_max =  np.nanmax(np.nanmean(respTraces_allTrials[iEpoch][:,:,:,:],axis=3),axis=0)
        curr_mean = np.nanmean(np.nanmean(respTraces_allTrials[iEpoch][:,:,:,:],axis=3),axis=0)
        
        maxResp[iEpoch] = curr_max
        MaxResp_matrix[:,:,index] = curr_max
        
        meanResp[iEpoch] = curr_mean
        MeanResp_matrix[:,:,index] = curr_mean
        
    
    # Make an additional one with edge and set edge maximum to 0 in the main array
    # This is to avoid assigning pixels to edge temporal frequency if they respond
    # max in the edge epoch but not in one of the grating epochs.
    MaxResp_matrix_all_epochs = copy.deepcopy(MaxResp_matrix)
    MeanResp_matrix_all_epochs = copy.deepcopy(MeanResp_matrix)

    
    # Finding pixel-wise max epochs
    maxEpochIdx_matrix_all = np.argmax(MaxResp_matrix_all_epochs,axis=2) 
    maxEpochIdx_matrix_all_mean = np.argmax(MeanResp_matrix_all_epochs,axis=2) 
    
    # To assign numbers like epoch numbers
    maxEpochIdx_matrix_all = maxEpochIdx_matrix_all + epoch_adjuster
    maxEpochIdx_matrix_all_mean = maxEpochIdx_matrix_all_mean + epoch_adjuster
    
    
    return MaxResp_matrix_all_epochs, maxEpochIdx_matrix_all, \
           MeanResp_matrix_all_epochs, maxEpochIdx_matrix_all_mean
            


def create_DSI_image(stimulus_information, maxEpochIdx_matrix_all,max_resp_matrix_all,
                     MaxResp_matrix_all_epochs):
    """ Makes pixel-wise plot of DSI

    Parameters
    ==========
   stimulus_information : list 
        Stimulus related information is stored here.
        
    maxEpochIdx_matrix_all : np array
        An array with pixel-wise maximum epoch indices. Adjusts the indices considering
        that baseline index is 0 and epochs start from 1.
        
    max_resp_matrix_all : np array
        An array with pixel-wise maxiumum responses for the experiment.
        
    MaxResp_matrix_all_epochs : np array
        An array with pixel-wise maxiumum responses for every epoch.
        
    

     
    Returns
    =======
    
    DSI_image : numpy.ndarray
        An image with CSI values ranging between -1 and 1. (-1 OFF 1 ON selective)
    
    """
    
    DSI_image = copy.deepcopy(max_resp_matrix_all) # copy it for keeping nan value
    for iEpoch, current_epoch_type in enumerate (stimulus_information['stim_type']):
        
        if (stimulus_information['random']) and (iEpoch ==0):
            continue
        
        current_pixels = (maxEpochIdx_matrix_all == iEpoch) & \
                            (~np.isnan(max_resp_matrix_all))
        current_freq = stimulus_information['epoch_frequency'][iEpoch]
        if ((current_epoch_type != 50) and (current_epoch_type != 61) and\
            (current_epoch_type != 46)) or (current_freq ==0):
            DSI_image[current_pixels] = 0
            continue
        current_dir = stimulus_information['epoch_dir'][iEpoch]
        required_epoch_array = \
            (stimulus_information['epoch_dir'] == ((current_dir+180) % 360)) & \
            (stimulus_information['epoch_frequency'] == current_freq) & \
            (stimulus_information['stim_type'] == current_epoch_type)
            
        opposite_dir_epoch = [epoch_indx for epoch_indx, epoch in \
                              enumerate(required_epoch_array) if epoch][0]
        opposite_dir_epoch = opposite_dir_epoch # To find the real epoch number without the baseline
        # Since a matrix will be indexed which doesn't have any information about the baseline epoch
        
    
        opposite_response_trace = MaxResp_matrix_all_epochs\
            [:,:,opposite_dir_epoch-stimulus_information['epoch_adjuster']] 
            
        DSI_image[current_pixels] =(np.abs(((max_resp_matrix_all[current_pixels] - \
                                      opposite_response_trace[current_pixels])\
                                        /(max_resp_matrix_all[current_pixels] + \
                                          opposite_response_trace[current_pixels]))))
#    DSI_image[(DSI_image>1)] = 0
#    DSI_image[(DSI_image<-1)] = 0
    
    return DSI_image

def create_CSI_image(stimulus_information, frameRate,respTraces_allTrials, 
                     DSI_image):
    """ Makes pixel-wise plot of CSI

    Parameters
    ==========
   stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    respTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
            
    DSI_image : numpy.ndarray
        An image with CSI values ranging between -1 and 1. (-1 OFF 1 ON selective)
    

     
    Returns
    =======
    
    CSI_image : numpy.ndarray
        An image with CSI values ranging between -1 and 1. (-1 OFF 1 ON selective)
    
    """
    # Image dimensions
    mov_xDim = np.shape(DSI_image)[0]
    mov_yDim = np.shape(DSI_image)[1]
    # Find edge epochs
    edge_epochs = np.where(stimulus_information['stim_type']==50)[0]
    epochDur= stimulus_information['epochs_duration']
    
    if len(edge_epochs) == 2: # 2 edges exist 
        
        ON_resp = np.zeros(shape=(mov_xDim,mov_yDim,2))
        OFF_resp = np.zeros(shape=(mov_xDim,mov_yDim,2))
        CSI_image = np.zeros(shape=(mov_xDim,mov_yDim))
        half_dur_frames = int((round(frameRate * epochDur[edge_epochs[0]]))/2)
        
        for index, epoch in enumerate(edge_epochs):
            
            
            OFF_resp[:,:,index] = np.nanmax(np.nanmean(\
                    respTraces_allTrials[epoch]\
                    [:half_dur_frames,:,:,:],axis=3),axis=0)
            ON_resp[:,:,index] = np.nanmax(np.nanmean(\
                   respTraces_allTrials[epoch]\
                   [half_dur_frames:,:,:,:],axis=3),axis=0)
        
        
        CSI_image[DSI_image>0] = (ON_resp[:,:,0][DSI_image>0] - OFF_resp[:,:,0][DSI_image>0])/(ON_resp[:,:,0][DSI_image>0] + OFF_resp[:,:,0][DSI_image>0])
        CSI_image[DSI_image<0] = (ON_resp[:,:,1][DSI_image<0] - OFF_resp[:,:,1][DSI_image<0])/(ON_resp[:,:,1][DSI_image<0] + OFF_resp[:,:,1][DSI_image<0])
        
    # It shouldn't be below -1 or above +1 if it is not noise
    CSI_image[(CSI_image>1)] = 0
    CSI_image[(CSI_image<-1)] = 0    
    
    return CSI_image


def plot_pixel_maps(im1, im2, im3, im4, exp_ID, depth, save_fig = False,
                    save_dir = None):

    """ Plots 4 images in a figure. Normally used with mean, max, DSI, CSI 
    images
    
    Parameters
    ==========
    mean_image : numpy array
        Mean image of the video.
    
    max_resp_matrix_all : numpy array
        Maximum responses
        
    DSI_image : numpy array
        Mean image of the video.
        
    CSI_image : numpy array
        Mean image of the video.
        
    exp_ID : str
    
    depth : int or float
        

    """
    plt.close('all')
    
    plt.style.use("dark_background")
    fig1, ax1 = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True,
                            facecolor='k', edgecolor='w',figsize=(16, 10))
    
    
    depthstr = 'Z: %d' % depth
    figtitle = 'Summary ' + depthstr
    
    # Mean image
    fig1.suptitle(figtitle,fontsize=12)
    sns.heatmap(im1,ax=ax1[0][0],cbar_kws={'label': 'dF/F'},cmap='viridis')
    #    sns.heatmap(layer_masks,alpha=.2,cmap='Set1',ax=ax2[0],cbar=False)
    #    sns.heatmap(BG_mask,alpha=.1,ax=ax2[0],cbar=False)
    ax1[0][0].axis('off')
    ax1[0][0].set_title('Mean image')
    
    # Max responses
    sns.heatmap(im2,cbar_kws={'label': 'SNR'},ax=ax1[0][1])
    #sns.heatmap(DSI_image,cbar_kws={'label': 'DSI'},cmap = 'RdBu_r',ax=ax2[1])
    ax1[0][1].axis('off')
    ax1[0][1].set_title('SNR image')
    # DSI
    sns.heatmap(im3,cbar_kws={'label': 'DSI'},cmap = 'inferno',ax=ax1[1][0])
    #sns.heatmap(DSI_image,cbar_kws={'label': 'DSI'},cmap = 'RdBu_r',ax=ax2[1])
    ax1[1][0].axis('off')
    ax1[1][0].set_title('DSI (Blue: --> Red: <--)')
    
    #CSI
    sns.heatmap(im4,cbar_kws={'label': 'CSI'},cmap = 'inferno',ax=ax1[1][1],vmax=1,vmin=-1)
    
    ax1[1][1].axis('off')    
    ax1[1][1].set_title('CSI (Dark:OFF, Red:ON)')
    
    if save_fig:
        # Saving figure
        save_name = 'summary_%s' % (exp_ID)
        os.chdir(save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
        print('Pixel maps saved')

def generate_avg_movie(dataDir, stimulus_information, 
                           wholeTraces_allTrials_video):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces.
    
    Parameters
    ==========
    dataDir: str
        Path into the directory where the motion corrected dataset with selected
        masks is present.
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
    wholeTraces_allTrials_video : list containing np arrays
        Epoch list of time traces including all trials in the form of:
        baseline epoch - stimulus epoch - baseline epoch
            
    
        
    Returns
    =======
    
    cluster_dataset : sima.imaging.ImagingDataset 
        Sima dataset to be used for segmentation.
    """
    print('Generating averaged movie...\n')
    # Directory for where to save the cluster movie
    selected_movie_dir = os.path.join(dataDir,'processed.sima')
    mov_xDim = np.shape(wholeTraces_allTrials_video[1])[1]
    mov_yDim = np.shape(wholeTraces_allTrials_video[1])[2]    
    
    epochs_to_use = range(len(stimulus_information['epoch_frequency']))
    if stimulus_information['random'] == 1:
        epochs_to_use = np.delete(epochs_to_use,stimulus_information['baseline_epoch'])
    epoch_frames = np.zeros(shape=np.shape(epochs_to_use))
    
    # Generating and filling the movie array movie array
    for index, epoch in enumerate(epochs_to_use):
        epoch_frames[index] = np.shape(wholeTraces_allTrials_video[epoch])[0]
        
    avg_movie = np.zeros(shape=(int(epoch_frames.sum()),1,mov_xDim,mov_yDim,1))
    
    startFrame = 0
    for index, epoch in enumerate(epochs_to_use):
        if index>0:
            startFrame =  endFrame 
        endFrame = startFrame + epoch_frames[index]
        avg_movie[int(startFrame):int(endFrame),0,:,:,0] = \
            wholeTraces_allTrials_video[epoch].mean(axis=3)
            
    
    # Create a sima dataset and export the cluster movie
    b = sima.Sequence.create('ndarray',avg_movie)
    average_dataset = sima.ImagingDataset([b],None)
    average_dataset.export_frames([[[os.path.join(selected_movie_dir,'avg_vid.tif')]]],
                                      fill_gaps=True,scale_values=True)
    print('Averaged movie generated...\n')
    
    return average_dataset


def find_clusters_STICA(cluster_dataset, area_min, area_max):
    """ Makes pixel-wise plot of DSI

    Parameters
    ==========
    cluster_dataset : sima.imaging.ImagingDataset 
        Sima dataset to be used for segmentation.
        
    area_min : int
        Minimum area of a cluster in pixels
        
    area_max : int
        Maximum area of a cluster in pixels
        
    

     
    Returns
    =======
    
    clusters : sima.ROI.ROIList
        A list of ROIs.
        
    all_clusters_image: numpy array
        A numpy array that contains the masks.
    """    
    print('\n-->Segmentation running...')
    segmentation_approach = sima.segment.STICA(channel = 0,components=45,mu=0.1)
    segmentation_approach.append(sima.segment.SparseROIsFromMasks(
            min_size=area_min,smooth_size=3))
    #segmentation_approach.append(sima.segment.MergeOverlapping(threshold=0.90))
    #segmentation_approach.append(sima.segment.SmoothROIBoundaries(tolerance=0.1,n_processes=(nCpu - 1)))
    size_filter = sima.segment.ROIFilter(lambda roi: roi.size >= area_min and \
                                         roi.size <= area_max)
#    circ_filter = sima.segment.CircularityFilter(circularity_threhold=0.7)
    segmentation_approach.append(size_filter)
#    segmentation_approach.append(circ_filter)
    start1 = time.time()
    clusters = cluster_dataset.segment(segmentation_approach, 'auto_ROIs')
    initial_cluster_num = len(clusters)
    end1 = time.time()
    time_passed = end1-start1
    print('Clusters found in %d minutes\n' % \
          round(time_passed/60) )
    print('Number of initial clusters: %d\n' % initial_cluster_num)
    
    
    # Generating an image with all clusters
    data_xDim = cluster_dataset.frame_shape[1]
    data_yDim = cluster_dataset.frame_shape[2]
    all_clusters_image = np.zeros(shape=(data_xDim,data_yDim))
    all_clusters_image[:] = np.nan
    for index, roi in enumerate(clusters):
        curr_mask = np.array(roi)[0,:,:]
        all_clusters_image[curr_mask] = index+1        
        
    return clusters, all_clusters_image

def get_layers_bg_mask(dataDir):
    """ Gets the masks of pre-selected (with roibuddy) layers.

    Parameters
    ==========
    dataDir: str
        Path into the directory where the motion corrected dataset with selected
        masks is present.
    
  
    Returns
    =======
    
    layer_masks_bool: np array
        A boolean image of where the masks are located
        
    BG_mask: np array
        A boolean image of background mask
    
    """                       
                                    
    dataset = sima.ImagingDataset.load(dataDir)
    roiKeys = dataset.ROIs.keys()
    roiKeyNo = 0
    rois_layer = dataset.ROIs[roiKeys[roiKeyNo]]
    layer_masks = np.zeros(shape=(np.shape(np.array(rois_layer[0]))[1],np.shape(np.array(rois_layer[0]))[2]))
    layer_masks_bool = np.zeros(shape=(np.shape(np.array(rois_layer[0]))[1],np.shape(np.array(rois_layer[0]))[2]))
    layer_masks[:] = np.nan
    
    BG_mask = np.zeros(shape=(np.shape(np.array(rois_layer[0]))[1],np.shape(np.array(rois_layer[0]))[2]))
    BG_mask[:] = np.nan
    for index, roi in enumerate(rois_layer):
        curr_mask = np.array(roi)[0,:,:]
        roi_label = roi.label
        
        if roi_label == 'Layer1':
            
            L1_mask = curr_mask
            layer_masks[curr_mask] = 1
            layer_masks_bool[curr_mask] = 1
            print("Layer 1 mask found\n")
        elif roi_label == 'Layer2':
            L2_mask = curr_mask
            layer_masks[curr_mask] = 2
            layer_masks_bool[curr_mask] = 1
            print("Layer 2 mask found\n")
        elif roi_label == 'Layer3':
             
            L3_mask = curr_mask
            layer_masks[curr_mask] = 3
            layer_masks_bool[curr_mask] = 1
            print("Layer 3 mask found\n")
        elif roi_label == 'Layer4':
            L4_mask = curr_mask
            layer_masks[curr_mask] = 4
            layer_masks_bool[curr_mask] = 1
            print("Layer 4 mask found\n")
        elif roi_label == 'LobDen':
            LD_mask = curr_mask
    #        layer_masks[curr_mask] = 3
    #        layer_masks_bool[curr_mask] = 1
        elif roi_label == 'MedDen':
            MD_mask = curr_mask
    #        layer_masks[curr_mask] = 4
    #        layer_masks_bool[curr_mask] = 1
        elif roi_label == 'BG':
            BG_mask = curr_mask
            print("BG mask found\n")
        else:
            print("ROI doesn't match to criteria: ")
            print(roi_label)      

    return layer_masks_bool, BG_mask                              
                                
                        
def select_regions(image_to_select_from, image_cmap ="gray",pause_t=7,
                   ask_name=True):
    """ Enables user to select rois from a given image using roipoly module.

    Parameters
    ==========
    image_to_select_from : numpy.ndarray
        An image to select ROIs from
    
    Returns
    =======
    
    """
    import warnings 
    plt.close('all')
    stopsignal = 0
    roi_number = 0
    roi_masks = []
    mask_names = []
    
    im_xDim = np.shape(image_to_select_from)[0]
    im_yDim = np.shape(image_to_select_from)[1]
    mask_agg = np.zeros(shape=(im_xDim,im_yDim))
    iROI = 0
    plt.style.use("dark_background")
    while (stopsignal==0):

        
        # Show the image
        fig = plt.figure()
        plt.imshow(image_to_select_from, interpolation='nearest', cmap=image_cmap)
        plt.colorbar()
        curr_agg = mask_agg.copy()
        curr_agg[curr_agg==0] = np.nan
        plt.imshow(curr_agg, alpha=0.3,cmap = 'Accent')
        plt.title("Select ROI: ROI%d" % roi_number)
        plt.show(block=False)
       
        
        # Draw ROI
        curr_roi = RoiPoly(color='r', fig=fig)
        iROI = iROI + 1
        # plt.waitforbuttonpress() # Used in Spyder only
        # plt.pause(pause_t) # Used in Spyder only
        if ask_name:
            mask_name = raw_input("\nEnter the ROI name:\n>> ")
        else:
            mask_name = iROI
        curr_mask = curr_roi.get_mask(image_to_select_from)
        if len(np.where(curr_mask)[0]) ==0 :
            warnings.warn('ROI empty.. discarded.') 
            continue
        mask_names.append(mask_name)
        
        
        roi_masks.append(curr_mask)
        
        mask_agg[curr_mask] += 1
        
        
        
        roi_number += 1
        signal = raw_input("\nPress k for exiting program, otherwise press enter")
        if (signal == 'k\r') or (signal == 'k'):
            stopsignal = 1
        
    
    return roi_masks, mask_names

def clusters_restrict_size_regions(rois, cluster_region_bool,
                                   cluster_1d_max_size_pixel,
                                   cluster_1d_min_size_pixel,
                                   otsu_thresholded_mask):
    """
    """    
    
    # Getting rid of clusters based on pre-defined regions and size
    passed_rois  = []
    ROI_mod.calcualte_mask_1d_size(rois)
    for roi in rois:
        # Check if mask is within the pre-defined regions
        mask_inclusion_points =  np.where(roi.mask * cluster_region_bool)[0]
        otsu_inclusion_points =  np.where(roi.mask * otsu_thresholded_mask)[0]
        
        if (mask_inclusion_points.size == np.where(roi.mask)[0].size) and\
            (otsu_inclusion_points.size>(roi.mask.sum()/2)): 
            # Check if mask is within the size restrictions
            if ((roi.x_size < cluster_1d_max_size_pixel) & (roi.y_size < cluster_1d_max_size_pixel) & \
                   (roi.x_size > cluster_1d_min_size_pixel) & (roi.y_size > cluster_1d_min_size_pixel)):
                
                passed_rois.append(roi)
                
            
    # Generating an image with masks
    data_xDim = np.shape(rois[0].mask)[0]
    data_yDim = np.shape(rois[0].mask)[1]
    
    passed_rois_image = np.zeros(shape=(data_xDim,data_yDim))
    passed_rois_image[:] = np.nan
    for index, roi in enumerate(passed_rois):
        passed_rois_image[roi.mask] = index+1
    
    print('Clusters excluded based on layers...')
    
    all_pre_selected_mask = np.zeros(shape=(data_xDim,data_yDim))

    pre_selected_roi_indices = np.arange(len(passed_rois))
    pre_selected_roi_indices_copy = np.arange(len(passed_rois))
    
    for index, roi in enumerate(passed_rois):
        all_pre_selected_mask[roi.mask] += 1
        
    # Getting rid of overlapping clusters
    while len(np.where(all_pre_selected_mask>1)[0]) != 0:
        
        for index, roi_idx in enumerate(pre_selected_roi_indices):
            
            if pre_selected_roi_indices[index] != -1:
                curr_mask = passed_rois[roi_idx].mask
                non_intersection_matrix = \
                    (all_pre_selected_mask[curr_mask] == 1)
                
                if len(np.where(non_intersection_matrix)[0]) == 0: 
                    # get rid of cluster if it doesn't have any non overlapping part
                    pre_selected_roi_indices[index] = -1
                    all_pre_selected_mask[curr_mask] -= 1
                    
                elif (len(np.where(non_intersection_matrix)[0]) != len(all_pre_selected_mask[curr_mask])): 
                    # get rid of cluster if it has any overlapping part
                    pre_selected_roi_indices[index] = -1
                    all_pre_selected_mask[curr_mask] -= 1
            else:
               continue
    
    # To retrieve some clusters if there are no overlaps 
    for iRep in range(100):
        
        for index, roi in enumerate(pre_selected_roi_indices):
            if pre_selected_roi_indices[index] == -1:
        #        print(index)
                curr_mask = passed_rois[pre_selected_roi_indices_copy[index]].mask
                non_intersection_matrix = (all_pre_selected_mask[curr_mask] == 0)
                if (len(np.where(non_intersection_matrix)[0]) == len(all_pre_selected_mask[curr_mask])):
                    # If there's no cluster here add the cluster back
                    print('cluster added back')
                    pre_selected_roi_indices[index] = pre_selected_roi_indices_copy[index]
                    all_pre_selected_mask[curr_mask] += 1
    
    separated_roi_indices = pre_selected_roi_indices[pre_selected_roi_indices != -1]
    sep_masks_image = np.zeros(shape=(data_xDim,data_yDim))
    sep_masks_image[:] = np.nan
    separated_rois = []
    
    for index, sep_clus_idx in enumerate(separated_roi_indices):
        sep_masks_image[passed_rois[sep_clus_idx].mask] = index+1
        
        separated_rois.append(passed_rois[sep_clus_idx])
    
    print('Clusters separated...')
    print('Cluster pass ratio: %.2f' % (float(len(separated_rois))/\
                                        float(len(rois))))
    print('Total clusters: %d'% len(separated_rois))
    
    return separated_rois, sep_masks_image
        


def separate_trials_ROI_v3(time_series,rois,stimulus_information,
                           frameRate, df_method, df_use = True, plotting=False,
                           max_resp_trial_len = 'max'):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces. Adds responses and whole traces into the ROI_bg 
    instances.
    
    Parameters
    ==========
    time_series : numpy array
        Time series in the form of: frames x m x n (m & n are pixel dimensions)
    
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list.
        Each term in this list is a trial of the epoch. Trials consist of 
        previous baseline epoch _ stimulus epoch _ following baseline epoch
        (if there is a baseline presentation)
        These terms have the following str: [[X, Y], [Z, D]] where
        first term is the trial beginning (first of first) and end
        (second of first), and second term is the baseline start
        (first of second) and end (second of second) for that trial.
    
    rois : list
        A list of ROI_bg instances.
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    df_method : str
        Method for calculating dF/F defined in the ROI_bg class.
        
    plotting: bool
        If the user wants to visualize the masks and the traces for clusters.
        
    Returns
    =======
    wholeTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            baseline epoch - stimulus epoch - baseline epoch
            
    respTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-
            
    """
    wholeTraces_allTrials_ROIs = {}
    respTraces_allTrials_ROIs = {}
    baselineTraces_allTrials_ROIs = {}
    all_clusters_dF_whole_trace = np.zeros(shape=(len(rois),
                                                  np.shape(time_series)[0]))
    
    # dF/F calculation
    for iROI, roi in enumerate(rois):
        roi.raw_trace = time_series[:,roi.mask].mean(axis=1)
        roi.calculateDf(method=df_method,moving_avg = True, bins = 3)
        all_clusters_dF_whole_trace[iROI,:] = roi.df_trace
        
        if df_use:
            roi.base_dur = [] # Initialize baseline duration here (not good practice...)
        if plotting:
            plt.figure(figsize=(8, 7))
            grid = plt.GridSpec(8, 1, wspace=0.4, hspace=0.3)
            plt.subplot(grid[:7,0])
            roi.showRoiMask()
            plt.subplot(grid[7:8,0])
            plt.plot(roi.df_trace)
            plt.title('Cluster %d %s:' % (iROI, roi))
            
            
            plt.waitforbuttonpress()
            plt.close('all')
            
    trialCoor = stimulus_information['trial_coordinates']
    for iEpoch in trialCoor:
        currentEpoch = trialCoor[iEpoch]
        current_epoch_dur = stimulus_information['epochs_duration'][iEpoch]
        trial_numbers = len(currentEpoch)
        trial_lens = []
        resp_lens = []
        base_lens = []
        for curr_trial_coor in currentEpoch:
            current_trial_length = curr_trial_coor[0][1]-curr_trial_coor[0][0]
            trial_lens.append(current_trial_length)
            
            baselineStart = curr_trial_coor[1][0]
            baselineEnd = curr_trial_coor[1][1]
            base_len = baselineEnd - baselineStart
            
            base_lens.append(base_len) 
            
            resp_start = curr_trial_coor[0][0]+base_len
            resp_end = curr_trial_coor[0][1]-base_len
            resp_lens.append(resp_end-resp_start)
        
        trial_len =  min(trial_lens)
        resp_len = min(resp_lens)
        base_len = min(base_lens)
        
        if not((max_resp_trial_len == 'max') or \
               (current_epoch_dur < max_resp_trial_len)):
            resp_len = int(round(frameRate * max_resp_trial_len))+1
            
        wholeTraces_allTrials_ROIs[iEpoch] = {}
        respTraces_allTrials_ROIs[iEpoch] = {}
        baselineTraces_allTrials_ROIs[iEpoch] = {}
   
        for iCluster, roi in enumerate(rois):
            
            if stimulus_information['random']:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(resp_len,
                                                         trial_numbers))
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(base_len,
                                                         trial_numbers))
            else:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                base_len  = np.shape(wholeTraces_allTrials_ROIs\
                                     [stimulus_information['baseline_epoch']]\
                                     [iCluster])[0]
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(int(frameRate*1.5),
                                   trial_numbers))
            
            for trial_num , current_trial_coor in enumerate(currentEpoch):
                
                if stimulus_information['random']:
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    baselineStart = current_trial_coor[1][0]
                    baselineEnd = current_trial_coor[1][1]
                    
                    respStart = current_trial_coor[1][1]
                    epochEnd = current_trial_coor[0][1]
                    
                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                        roi_resp = roi.df_trace[respStart:epochEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        roi_resp = roi.raw_trace[respStart:epochEnd]
                    
                            
                    wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_resp[:resp_len]
                    baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:base_len]
                else:
                    
                    # If the sequence is non random  the trials are just separated without any baseline
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        
                    
                    if iEpoch == stimulus_information['baseline_epoch']:
                        baseline_trace = roi_whole_trace[:base_len]
                        baseline_trace = baseline_trace[-int(frameRate*1.5):]
                        baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= baseline_trace
                    else:
                        baselineTraces_allTrials_ROIs[iEpoch][iCluster]\
                            [:,trial_num]= baselineTraces_allTrials_ROIs\
                            [stimulus_information['baseline_epoch']][iCluster]\
                            [:,trial_num]
                    
                    wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    
                    
    for iEpoch in trialCoor:
        for iCluster, roi in enumerate(rois):
            
            # Appending trial averaged responses to roi instances only if 
            # df is used
            if df_use:
                if not stimulus_information['random']:
                    if iEpoch > 0 and iEpoch < len(trialCoor)-1:
                        
                        wt = np.concatenate((wholeTraces_allTrials_ROIs[iEpoch-1][iCluster].mean(axis=1),
                                            wholeTraces_allTrials_ROIs[iEpoch][iCluster].mean(axis=1),
                                            wholeTraces_allTrials_ROIs[iEpoch+1][iCluster].mean(axis=1)),
                                            axis =0)
                        roi.base_dur.append(len(wholeTraces_allTrials_ROIs[iEpoch-1][iCluster].mean(axis=1))) 
                    else:
                        wt = wholeTraces_allTrials_ROIs[iEpoch][iCluster].mean(axis=1)
                        roi.base_dur.append(0) 
                else:
                    wt = wholeTraces_allTrials_ROIs[iEpoch][iCluster].mean(axis=1)
                    base_dur = frameRate * stimulus_information['baseline_duration']
                    roi.base_dur.append(int(round(base_dur)))
                    
                roi.appendTrace(wt,iEpoch, trace_type = 'whole')
                roi.appendTrace(respTraces_allTrials_ROIs[iEpoch][iCluster].mean(axis=1),
                                  iEpoch, trace_type = 'response' )
                    
                    
                
        
    if df_use:
        print('Trial separation for ROIs completed')
    else:
        print('Trial separation not done (df not calculated)')
    return (wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs, 
            baselineTraces_allTrials_ROIs, all_clusters_dF_whole_trace) 

def separate_trials_ROI_v4(time_series,rois,stimulus_information,frameRate, 
                           df_method = None, df_use = True,
                           max_resp_trial_len = 'max'):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces. Adds responses and whole traces into the ROI_bg 
    instances.
    
    Parameters
    ==========
    time_series : numpy array
        Time series in the form of: frames x m x n (m & n are pixel dimensions)
    
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list.
        Each term in this list is a trial of the epoch. Trials consist of 
        previous baseline epoch _ stimulus epoch _ following baseline epoch
        (if there is a baseline presentation)
        These terms have the following str: [[X, Y], [Z, D]] where
        first term is the trial beginning (first of first) and end
        (second of first), and second term is the baseline start
        (first of second) and end (second of second) for that trial.
    
    rois : list
        A list of ROI_bg instances.
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    df_method : str
        Method for calculating dF/F defined in the ROI_bg class.
        
    plotting: bool
        If the user wants to visualize the masks and the traces for clusters.
        
    Returns
    =======
    wholeTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            baseline epoch - stimulus epoch - baseline epoch
            
    respTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-
            
    """
    wholeTraces_allTrials_ROIs = {}
    respTraces_allTrials_ROIs = {}
    baselineTraces_allTrials_ROIs = {}
    
    # dF/F calculation
    for iROI, roi in enumerate(rois):
        roi.raw_trace = time_series[:,roi.mask].mean(axis=1)
        roi.calculateDf(method=df_method,moving_avg = True, bins = 3)
        if df_use:
            roi.base_dur = [] # Initialize baseline duration here for upcoming analysis
            
    trialCoor = stimulus_information['trial_coordinates']
    # Trial averaging by loooping through epochs and trials
    for iEpoch in trialCoor:
        currentEpoch = trialCoor[iEpoch]
        current_epoch_dur = stimulus_information['epochs_duration'][iEpoch]
        trial_numbers = len(currentEpoch)
        trial_lens = []
        resp_lens = []
        base_lens = []
        for curr_trial_coor in currentEpoch:
            current_trial_length = curr_trial_coor[0][1]-curr_trial_coor[0][0]
            trial_lens.append(current_trial_length)
            
            baselineStart = curr_trial_coor[1][0]
            baselineEnd = curr_trial_coor[1][1]
            base_len = baselineEnd - baselineStart
            
            base_lens.append(base_len) 
            
            resp_start = curr_trial_coor[0][0]+base_len
            resp_end = curr_trial_coor[0][1]-base_len
            resp_lens.append(resp_end-resp_start)
        
        trial_len =  min(trial_lens)
        resp_len = min(resp_lens)
        base_len = min(base_lens)
        
        if not((max_resp_trial_len == 'max') or \
               (current_epoch_dur < max_resp_trial_len)):
            resp_len = int(round(frameRate * max_resp_trial_len))+1
            
        wholeTraces_allTrials_ROIs[iEpoch] = {}
        respTraces_allTrials_ROIs[iEpoch] = {}
        baselineTraces_allTrials_ROIs[iEpoch] = {}
   
        for iCluster, roi in enumerate(rois):
            
            # Baseline epoch is presented only when random value = 0 and 1 
            if stimulus_information['random'] == 1:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(resp_len,
                                                         trial_numbers))
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(base_len,
                                                         trial_numbers))
            elif stimulus_information['random'] == 0:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                base_len  = np.shape(wholeTraces_allTrials_ROIs\
                                     [stimulus_information['baseline_epoch']]\
                                     [iCluster])[0]
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(int(frameRate*1.5),
                                   trial_numbers))
            else:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = None
            
            for trial_num , current_trial_coor in enumerate(currentEpoch):
                
                if stimulus_information['random'] == 1:
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    baselineStart = current_trial_coor[1][0]
                    baselineEnd = current_trial_coor[1][1]
                    
                    respStart = current_trial_coor[1][1]
                    epochEnd = current_trial_coor[0][1]
                    
                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                        roi_resp = roi.df_trace[respStart:epochEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        roi_resp = roi.raw_trace[respStart:epochEnd]
                    try:
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    except ValueError:
                        new_trace = np.full((trial_len,),np.nan)
                        new_trace[:len(roi_whole_trace)] = roi_whole_trace.copy()
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                            
                    respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_resp[:resp_len]
                    baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:base_len]
                elif stimulus_information['random'] == 0:
                    
                    # If the sequence is non random  the trials are just separated without any baseline
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        
                    
                    if iEpoch == stimulus_information['baseline_epoch']:
                        baseline_trace = roi_whole_trace[:base_len]
                        baseline_trace = baseline_trace[-int(frameRate*1.5):]
                        baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= baseline_trace
                    else:
                        baselineTraces_allTrials_ROIs[iEpoch][iCluster]\
                            [:,trial_num]= baselineTraces_allTrials_ROIs\
                            [stimulus_information['baseline_epoch']][iCluster]\
                            [:,trial_num]
                    
                    try:
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    except ValueError:
                        new_trace = np.full((trial_len,),np.nan)
                        new_trace[:len(roi_whole_trace)] = roi_whole_trace.copy()
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                        
                else:
                    # If the sequence is all random the trials are just separated without any baseline
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                    
                    try:
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    except ValueError:
                        new_trace = np.full((trial_len,),np.nan)
                        new_trace[:len(roi_whole_trace)] = roi_whole_trace.copy()
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                    
    for iEpoch in trialCoor:
        for iCluster, roi in enumerate(rois):
            
            # Appending trial averaged responses to roi instances only if 
            # df is used
            if df_use:
                if stimulus_information['random'] == 0:
                    if iEpoch > 0 and iEpoch < len(trialCoor)-1:
                        
                        wt = np.concatenate((np.nanmean(wholeTraces_allTrials_ROIs[iEpoch-1][iCluster],axis=1),
                                            np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1),
                                            np.nanmean(wholeTraces_allTrials_ROIs[iEpoch+1][iCluster],axis=1)),
                                            axis =0)
                        roi.base_dur.append(len(np.nanmean(wholeTraces_allTrials_ROIs[iEpoch-1][iCluster],axis=1)))
                    else:
                        wt = np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1)
                        roi.base_dur.append(0) 
                elif stimulus_information['random'] == 1:
                    wt = np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1)
                    base_dur = frameRate * stimulus_information['baseline_duration']
                    roi.base_dur.append(int(round(base_dur)))
                else:
                    wt = np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1)
                    
                    
                roi.appendTrace(wt,iEpoch, trace_type = 'whole')
                roi.appendTrace(np.nanmean(respTraces_allTrials_ROIs[iEpoch][iCluster],axis=1),
                                  iEpoch, trace_type = 'response' )
                    
                    
                
        
    if df_use:
        print('Traces are stored in ROI objects.')
    else:
        print('No trace is stored in objects.')
    return (wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs, 
            baselineTraces_allTrials_ROIs)                 
        
def plot_roi_properties(images, properties, colormaps,underlying_image,vminmax,
                        exp_ID,depth,save_fig = False, save_dir = None,
                        figsize=(10, 6),alpha=0.5):
    """ 
    Parameters
    ==========
    
        
    Returns
    =======

    """
    plt.close('all')
    run_matplotlib_params()    
    plt.style.use('dark_background')
    total_n_images = len(images)
    col_row_n = math.ceil(math.sqrt(total_n_images))
    
    fig1, ax1 = plt.subplots(ncols=col_row_n, nrows=col_row_n, sharex=True, 
                             sharey=True,figsize=figsize)
    depthstr = 'Z: %d' % depth
    figtitle = 'ROIs summary: ' + depthstr
    fig1.suptitle(figtitle,fontsize=12)
    
    for idx, ax in enumerate(ax1.reshape(-1)): 
        if idx >= total_n_images:
            ax.axis('off')
        else:
            sns.heatmap(underlying_image,cmap='gray',ax=ax,cbar=False)
            
            sns.heatmap(images[idx],alpha=alpha,cmap = colormaps[idx],ax=ax,
                        cbar=True,cbar_kws={'label': properties[idx]},
                        vmin = vminmax[idx][0], vmax=vminmax[idx][1])
            ax.axis('off')
    
    if save_fig:
        # Saving figure
        save_name = 'ROI_props_%s' % (exp_ID)
        os.chdir(save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight',dpi=300)
        print('ROI property images saved')
        

def plot_roi_masks(roi_image, underlying_image,n_roi1,exp_ID,
                       save_fig = False, save_dir = None,alpha=0.5):
    """ Plots two different cluster images underlying an another common image.
    Parameters
    ==========
    first_clusters_image : numpy array
        An image array where clusters (all from segmentation) have different 
        values.
    
    second_cluster_image : numpy array
        An image array where clusters (the final ones) have different values.
        
    underlying_image : numpy array
        An image which will be underlying the clusters.
        
    Returns
    =======

    """

    plt.close('all')
    plt.style.use("dark_background")
    fig1, ax1 = plt.subplots(ncols=1, nrows=1,facecolor='k', edgecolor='w',
                             figsize=(5, 5))
    
    # All clusters
    sns.heatmap(underlying_image,cmap='gray',ax=ax1,cbar=False)
    sns.heatmap(roi_image,alpha=alpha,cmap = 'tab20b',ax=ax1,
                cbar=False)
    
    ax1.axis('off')
    ax1.set_title('ROIs n=%d' % n_roi1)
    
    if save_fig:
        # Saving figure
        save_name = 'ROIs_%s' % (exp_ID)
        os.chdir(save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
        print('ROI images saved')


def plot_raw_responses_stim(responses, rawStimData, exp_ID, save_fig =False, 
                            save_dir = None, ax_to_plot =None):
    """ Gets the required stimulus and imaging parameters.
    Parameters
    ==========
    responses : n x m numpy array
        Response traces along the row dimension. (n ROIs, m time points)
    
    rawStimData : numpy array
        Raw stimulus output data where the frames and stim values are stored.
        
    Returns
    =======
    

    """
    
    adder = np.linspace(0, np.shape(responses)[0]*4, 
                        np.shape(responses)[0])[:,None]
    scaled_responses = responses + adder
    
    # Finding stimulus
    
    stim_frames = rawStimData[:,7]  # Frame information
    stim_vals = rawStimData[:,3] # Stimulus value
    uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
    stim_vals = stim_vals[uniq_frame_id]
    # Make normalized values of stimulus values for plotting
    
    stim_vals = (stim_vals/np.max(np.unique(stim_vals))) \
        *np.max(scaled_responses)/3
    stim_df = pd.DataFrame(stim_vals,columns=['Stimulus'],dtype='float')
    
    resp_df = pd.DataFrame(np.transpose(scaled_responses),dtype='float')
    
    if ax_to_plot is None:
        ax = resp_df.plot(legend=False,alpha=0.8,lw=0.5)
    else:
        ax = ax_to_plot
        resp_df.plot(legend=False,alpha=0.8,lw=0.5,ax=ax_to_plot)
        
    stim_df.plot(dashes=[2, 1],ax=ax,color='w',alpha=.8,lw=2)
    plt.title('Responses (N:%d)' % np.shape(responses)[0])
    
    if save_fig:
        # Saving figure
        save_name = 'ROI_traces%s' % (exp_ID)
        os.chdir(save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
        print('All traces figure saved')
    

def calculate_SNR_Corr(base_traces_all_roi, resp_traces_all_roi,
                       rois, epoch_to_exclude = None):
    """ Calculates the signal-to-noise ratio (SNR). Equation taken from
    Kouvalainen et al. 1994 (see calculation of SNR true from SNR estimated).
    Also calculates the correlation between the first and the last trial to 
    estimate the reliability of responses.
    
    
    Parameters
    ==========
    respTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-
            
    rois : list
        A list of ROI_bg instances.
        
    epoch_to_exclude : int 
        Default: None
        Epoch number to exclude when calculating corr and SNR
        
        
    Returns
    =======
    
    SNR_max_matrix : np array
        SNR values for all ROIs.
        
    Corr_matrix : np array
        SNR values for all ROIs.
        
    """
    total_epoch_numbers = len(base_traces_all_roi)
    
    SNR_matrix = np.zeros(shape=(len(rois),total_epoch_numbers))
    Corr_matrix = np.zeros(shape=(len(rois),total_epoch_numbers))
    
    for iROI, roi in enumerate(rois):
        
        for iEpoch, iEpoch_index in enumerate(base_traces_all_roi):
            
            if iEpoch_index == epoch_to_exclude:
                SNR_matrix[iROI,iEpoch] = 0
                Corr_matrix[iROI,iEpoch] = 0
                continue
            
            trial_numbers = np.shape(resp_traces_all_roi[iEpoch_index][iROI])[1]
            
            
            currentBaseTrace = base_traces_all_roi[iEpoch_index][iROI][:,:]
            currentRespTrace =  resp_traces_all_roi[iEpoch_index][iROI][:,:]
            
            # Reliability between all possible combinations of trials
            perm = permutations(range(trial_numbers), 2) 
            coeff =[]
            for iPerm, pair in enumerate(perm):
                curr_coeff, pval = pearsonr(currentRespTrace[:-2,pair[0]],
                                            currentRespTrace[:-2,pair[1]])
                coeff.append(curr_coeff)
                
            coeff = np.array(coeff).mean()
            
            noise_std = currentBaseTrace.std(axis=0).mean(axis=0)
            resp_std = currentRespTrace.std(axis=0).mean(axis=0)
            signal_std = resp_std - noise_std
            # SNR calculation taken from
            curr_SNR_true = ((trial_numbers+1)/trial_numbers)*(signal_std/noise_std) - 1/trial_numbers
        #        curr_SNR = (signal_std/noise_std) 
            SNR_matrix[iROI,iEpoch] = curr_SNR_true
            Corr_matrix[iROI,iEpoch] = coeff
        
        roi.SNR = np.nanmax(SNR_matrix[iROI,:])
        roi.reliability = np.nanmax(Corr_matrix[iROI,:])
    
     
    SNR_max_matrix = np.nanmax(SNR_matrix,axis=1) 
    Corr_matrix = np.nanmax(Corr_matrix,axis=1)
    
    return SNR_max_matrix, Corr_matrix



def plot_df_dataset(df, properties, save_name = 'ROI_plots_%s', 
                    exp_ID=None, save_fig = False, save_dir=None):
    """ Plots a variable against 3 other variables

    Parameters
    ==========
   
     
    Returns
    =======
    
    
    """
    plt.close('all')
    colors = run_matplotlib_params()    
    if len(properties) < 5:
        dim1= len(properties)
        dim2 = 1
    elif len(properties)/5.0 >= 1.0:
        dim1 = 5
        dim2 = int(np.ceil(len(properties)/5.0))
        
    fig1, ax1 = plt.subplots(ncols=dim1, nrows=dim2,figsize=(10, 3))
    axs = ax1.flatten()
    
    for idx, ax in enumerate(axs):
        try:
            sns.distplot(df[properties[idx]],ax=ax,color=plt.cm.Dark2(3),rug=True,
                         hist=False)
        except:
            continue
    
    if save_fig:
            # Saving figure
            save_name = 'ROI_plots_%s' % (exp_ID)
            os.chdir(save_dir)
            plt.savefig('%s.png'% save_name, bbox_inches='tight',dpi=300)
            print('ROI properties saved')
    return None


def run_roi_transfer(transfer_data_path, transfer_type,experiment_info=None,
                     imaging_info=None):
    '''
    
    Updates:
        25/03/2020 - Removed transfer types of 11 steps and AB steps since they
        are redundant with minimal type
    '''
    load_path = open(transfer_data_path, 'rb')
    workspace = pickle.load(load_path)
    rois = workspace['final_rois']
    
    if transfer_type == 'luminance_gratings' or \
        transfer_type == 'lum_con_gratings' :
        
        properties = ['CSI', 'CS','PD','DSI','category',
                      'analysis_params']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
            
    elif transfer_type == 'stripes_OFF_delay_profile':
        
        properties = ['CSI', 'CS','PD','DSI','two_d_edge_profile','category',
                      'analysis_params','edge_start_loc','edge_speed']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,CS='OFF')
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
        
    elif transfer_type == 'stripes_ON_delay_profile':
        properties = ['CSI', 'CS','PD','DSI','two_d_edge_profile','category',
                      'analysis_params','edge_start_loc','edge_speed']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,CS='ON')

        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))

    elif ((transfer_type == 'stripes_ON_vertRF_transfer') or \
          (transfer_type == 'stripes_ON_horRF_transfer') or \
          (transfer_type == 'stripes_OFF_vertRF_transfer') or \
          (transfer_type == 'stripes_OFF_horRF_transfer')):
        properties = ['corr_fff', 'max_response','category','analysis_params']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    elif (transfer_type == 'ternaryWN_elavation_RF'):
        properties = ['corr_fff', 'max_response','category','analysis_params',
                      'reliability','SNR']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    elif (transfer_type == 'gratings_transfer_rois_save'):
        
        properties = ['CSI', 'CS','PD','DSI','category','RF_maps','RF_map',
                      'RF_center_coords','analysis_params','RF_map_norm']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        
    elif (transfer_type == 'luminance_edges_OFF' ):
        if (('R64G09' in rois[0].experiment_info['Genotype']) or \
         ('T5' in rois[0].experiment_info['Genotype'])):
            CS = 'OFF'
            warnings.warn('Transferring only T5 neurons')
        else:
            CS = None
            warnings.warn('NO CS selected since genotype is not found to be T4-5')
        
        properties = ['CSI', 'CS','PD','DSI','category','RF_maps','RF_map',
                      'RF_center_coords','analysis_params','RF_map_norm']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,CS=CS)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    elif (transfer_type == 'luminance_edges_ON'):
        
        if (('R64G09' in rois[0].experiment_info['Genotype']) or \
         ('T4' in rois[0].experiment_info['Genotype'])):
            CS = 'ON'
            warnings.warn('Transferring only T4 neurons')
        else:
            CS = None
            warnings.warn('NO CS selected since genotype is not found to be T4-5')
        properties = ['CSI', 'CS','PD','DSI','category','RF_maps','RF_map',
                      'RF_center_coords','analysis_params','RF_map_norm']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,CS=CS)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
            
   
    elif transfer_type == 'STF_1':
        properties = ['CSI', 'CS','PD','DSI','category','analysis_params']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
            
    elif transfer_type == 'minimal' :
        print('Transfer type is minimal... Transferring just masks, categories and if present RF maps...\n')
        properties = ['category','analysis_params','RF_maps','RF_map',
                      'RF_center_coords','RF_map_norm']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    else:
        raise NameError('Invalid ROI transfer type')
        
        
   
    return transferred_rois

def run_analysis(analysis_params, rois,experiment_conditions,
                 imaging_information,summary_save_dir,
                 save_fig=True,fig_save_dir = None, 
                 exp_ID=None):
    """
    asd
    """
    analysis_type = analysis_params['analysis_type']
    figtitle = 'Summary: %s Gen: %s | Age: %s | Z: %d' % \
           (experiment_conditions['MovieID'].split('-')[0],
            experiment_conditions['Genotype'], experiment_conditions['Age'],
            imaging_information['depth'])
    
    if analysis_type == 'A_B_steps':
        
        rois = ROI_mod.analyze_A_B_step(rois,int_rate = 10)
        
        if ("3OFF" not in rois[0].stim_name) and \
            ("varyingDur" not in rois[0].stim_name):
            roi_image = ROI_mod.get_masks_image(rois)
            fig, fig2 = sf.make_exp_summary_AB_steps(figtitle,rois,
                                                      roi_image,
                                                      experiment_conditions['MovieID'],
                                                      summary_save_dir)
            f_n = 'Summary_%s' % (exp_ID)
            os.chdir(fig_save_dir)
            fig.savefig('%s.png'% f_n, bbox_inches='tight',
                       transparent=False,dpi=300)
            
            f_n = 'Summary_%s_lum_con' % (exp_ID)
            os.chdir(fig_save_dir)
            fig2.savefig('%s.png'% f_n, bbox_inches='tight',
                       transparent=False,dpi=300)
    
        
    elif analysis_type == 'Flash_Steps':
        rois = ROI_mod.analyze_contrast_flashes(rois,int_rate = 10)

        contrasts = rois[0].epoch_contrasts[1:]
        resps = np.array(map(lambda roi: roi.contrast_responses[1:], rois))

        err = np.nanstd(resps,axis=0)/np.sqrt(np.shape(resps)[0])
        plt.scatter(np.tile(contrasts,(resps.shape[0],1)),resps,alpha=.5,s=15,color='w')
        plt.errorbar(contrasts,np.nanmean(resps,axis=0), err,fmt='s')
        
        plt.xlabel('Contrast')
        plt.ylabel('$\Delta F/F$')

        f_n = 'Summary_%s' % (exp_ID)
        fig = plt.gcf()
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f_n, bbox_inches='tight',
                   transparent=False,dpi=300)

    elif analysis_type == 'luminance_steps':
        
        rois = ROI_mod.analyze_luminance_steps(rois,int_rate = 10)
        roi_image = ROI_mod.get_masks_image(rois)
        fig = sf.make_exp_summary_luminance_steps(figtitle,rois,
                                                  roi_image,
                                                  experiment_conditions['MovieID'],
                                                  summary_save_dir)
        f_n = 'Summary_%s' % (exp_ID)
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f_n, bbox_inches='tight',
                   transparent=False,dpi=300)
        
    elif analysis_type == 'luminance_gratings':
        
        if ('T4' in rois[0].experiment_info['Genotype'] or \
            'T5' in rois[0].experiment_info['Genotype']) and \
            (not('1D' in rois[0].stim_name)):
            map(lambda roi: roi.calculate_DSI_PD(method='PDND'), rois)
            
        rois = ROI_mod.analyze_luminance_gratings(rois)
        run_matplotlib_params()
        mean_TFL = np.mean([np.array(roi.tfl_map) for roi in rois],axis=0)
        fig = plt.figure(figsize = (5,5))
        
        ax=sns.heatmap(mean_TFL, cmap='coolwarm',center=0,
                       xticklabels=np.array(rois[0].tfl_map.columns.levels[1]).astype(float),
                       yticklabels=np.array(rois[0].tfl_map.index),
                       cbar_kws={'label': '$\Delta F/F$'})
        ax.invert_yaxis()
        plt.title('TFL map')
        
        fig = plt.gcf()
        f0_n = 'Summary_TFL_%s' % (exp_ID)
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f0_n, bbox_inches='tight',
                    transparent=False,dpi=300)
        
    elif analysis_type == 'lum_con_gratings':
        run_matplotlib_params()
        rois = ROI_mod.analyze_gratings_general(rois)
        for roi in rois:
            roi_dict = {}
            roi_dict['Luminance'] = roi.luminances
            roi_dict['Contrast'] = roi.contrasts
            roi_dict['Response'] = roi.power_at_hz
            
            df_roi = pd.DataFrame.from_dict(roi_dict)
            cl_map = df_roi.pivot(index='Contrast',columns='Luminance')
            roi.cl_map= cl_map
            roi.cl_map_norm=(cl_map-cl_map.mean())/cl_map.std()

        mean_CL = np.mean([np.array(roi.cl_map) for roi in rois],axis=0)
        fig = plt.figure(figsize = (5,5))
        
        ax=sns.heatmap(mean_CL, cmap='coolwarm',center=0,
                       xticklabels=np.array(rois[0].cl_map.columns.levels[1]).astype(float),
                       yticklabels=np.array(rois[0].cl_map.index),
                       cbar_kws={'label': 'Response'})
        ax.invert_yaxis()
        plt.title('CL map')
        plt.xlabel('Luminance')
        plt.ylabel('Contrast')

        fig = plt.gcf()
        f0_n = 'Summary_CL_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f0_n, bbox_inches='tight',
                    transparent=False)


    elif analysis_type == '8D_10dps_stripes_RF':
        
        rois = ROI_mod.map_RF_adjust_stripe_time(rois,screen_props = {'45':74, '135':72,
                                                   '225':74,'315':72,
                                                   '0':53,'180':53,
                                                   '90':78,'270':78},
                                               delay_use=False)
    
        # random.shuffle(rois_plot)
        
        fig1 = ROI_mod.plot_RFs(rois, number=len(rois), f_w =5,cmap='coolwarm',
                                center_plot = True, center_val = 0.95)
        
        fig2 = ROI_mod.plot_RF_centers_on_screen(rois,prop=None, cmap='tab20b',
                              ylab='ROI num',lims=(1,len(rois)))
        fig3 = ROI_mod.plot_RF(rois[random.randint(0,len(rois)-1)],
                                   cmap1='coolwarm',cmap2='inferno')
        f3_n = 'BT_roi_example_%s' % (exp_ID)
        fig3.savefig('%s.png'% f3_n, bbox_inches='tight',
                   transparent=False,dpi=300)
                
        
        if save_fig:
           # Saving figure 
           f1_n = 'RF_examples_%s' % (exp_ID)
           f2_n = 'RF_on_screen_%s' % (exp_ID)
           
           os.chdir(fig_save_dir)
           fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
                       transparent=False,dpi=300)
           fig2.savefig('%s.png'% f2_n, bbox_inches='tight',
                       transparent=False,dpi=300)
           
    
    elif (analysis_type == '2D_edges_find_rois_delay_profile_save') or\
        (analysis_type == '2D_edges_find_save'):
        map(lambda roi: roi.calculate_DSI_PD(method='PDND'), rois)
        map(lambda roi: roi.calculate_CSI(frameRate=
                                          imaging_information['frame_rate']), 
            rois)
        rois = ROI_mod.generate_time_delay_profile_2Dedges(rois)
        
    elif ((analysis_type == 'luminance_edges_OFF' ) or\
          (analysis_type == 'luminance_edges_ON' )) :
        
        if ('T4' in rois[0].experiment_info['Genotype'] or \
            'T5' in rois[0].experiment_info['Genotype'] or \
            'R64G09' in rois[0].experiment_info['Genotype']    ):
            map(lambda roi: roi.calculate_DSI_PD(method='PDND'), rois)
        rois = ROI_mod.analyze_luminance_edges(rois,int_rate = 10)
        roi_image = ROI_mod.generate_colorMasks_properties(rois, 'slope')
        fig = sf.make_exp_summary_luminance_edges(figtitle,rois,
                                                  roi_image,
                                                  experiment_conditions['MovieID'],
                                                  summary_save_dir)
        
        
        slope_data = ROI_mod.data_to_list(rois, ['slope'])['slope']
        rangecolor= np.max(np.abs([np.min(slope_data),np.max(slope_data)]))
        
        if 'RF_map' in rois[0].__dict__:
            fig2 = ROI_mod.plot_RF_centers_on_screen(rois,prop='slope',
                                                     cmap='PRGn',
                                                     ylab='Lum sensitivity',
                                                     lims=(-rangecolor,
                                                           rangecolor))
            f2_n = 'Slope_on_screen_%s' % (exp_ID)
            os.chdir(fig_save_dir)
            fig2.savefig('%s.png'% f2_n, bbox_inches='tight',
                           transparent=False,dpi=300)
        else:
            print('No RF found for the ROI.')
        
        f1_n = 'Summary_%s' % (exp_ID)
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f1_n, bbox_inches='tight',
                       transparent=False,dpi=300)
        
        
        
        
              
             
    elif analysis_type == '5sFFF_analyze_save':
        rois = ROI_mod.conc_traces(rois, interpolation = True, int_rate = 10)
        roi_traces = list(map(lambda roi: roi.df_trace, rois))
        roi_conc_traces = list(map(lambda roi: roi.conc_trace, rois))
        stim_trace  = rois[0].stim_trace
        raw_stim = rois[0].stim_info['output_data']
        fig = sf.make_exp_summary_FFF(figtitle,
                             rois[0].source_image,
                             ROI_mod.get_masks_image(rois),
                             roi_traces,raw_stim,stim_trace,
                             roi_conc_traces,save_fig,
                             experiment_conditions['MovieID'],
                             summary_save_dir)
        f1_n = '5sFFF_summary_%s' % (exp_ID)
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f1_n, bbox_inches='tight',
                       transparent=False,dpi=300)
        
    elif analysis_type == '8D_edges_find_rois_save':
        map(lambda roi: roi.calculate_DSI_PD(method='vector'), rois)
        map(lambda roi: roi.calculate_CSI(frameRate=imaging_information['frame_rate']), 
            rois)
        rois = ROI_mod.map_RF_adjust_edge_time(rois,edges=True,
                                               delay_degrees=9.6,
                                               delay_use=True,
                                               edge_props = {'45':71, 
                                                             '135':71,
                                                             '225':71,'315':71, 
                                                             '0':51,'180':51, 
                                                             '90':75,'270':75})
        copy_rois = copy.deepcopy(rois)
        plot_reliability_n = 0.7
        rois_plot = [roi for roi in copy_rois if roi.reliability>plot_reliability_n]
        # random.shuffle(rois_plot)
        
        fig1 = ROI_mod.plot_RFs(rois_plot, number=20, f_w =5,cmap='coolwarm',
                                center_plot = True, center_val = 0.95)
        
        fig2 = ROI_mod.plot_RF_centers_on_screen(rois,prop='PD')
        try:
            fig3 = ROI_mod.plot_RF(rois_plot[random.randint(0,len(rois_plot))],
                                   cmap1='coolwarm',cmap2='inferno')
            f3_n = 'BT_roi_example_%s' % (exp_ID)
            fig3.savefig('%s.png'% f3_n, bbox_inches='tight',
                       transparent=False,dpi=300)
        except:
            print('No roi above the reliability %.2f threshold' % plot_reliability_n)
                
        
        if save_fig:
           # Saving figure 
           f1_n = 'RF_examples_%s' % (exp_ID)
           f2_n = 'RF_on_screen_%s' % (exp_ID)
           
           os.chdir(fig_save_dir)
           fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
                       transparent=False,dpi=300)
           fig2.savefig('%s.png'% f2_n, bbox_inches='tight',
                       transparent=False,dpi=300)
           
           
    elif (analysis_type == 'stripes_OFF_delay_profile') or \
        (analysis_type == 'stripes_ON_delay_profile'):
        rois=ROI_mod.generate_time_delay_profile_combined(rois)
        
        
        if int(len(rois)/3) >10:
            f_w = 10
        else:
            f_w=int(len(rois)/3) 
            
        fig1 = ROI_mod.plot_delay_profile_examples(rois,number=None,f_w=None)
        
        if save_fig:
            # Saving figure 
            save_name = 'DelayProfiles_%s' % (exp_ID)
            os.chdir(fig_save_dir)
            fig1.savefig('%s.png'% save_name, bbox_inches='tight',
                        transparent=False)
        
        filt_rois = ROI_mod.filter_delay_profile_rois(rois,Rsq_t = 0)
        data_to_extract = ['resp_delay_deg', 'resp_delay_fits_Rsq','PD']
        filt_data = ROI_mod.data_to_list(filt_rois, data_to_extract)
        mean_rsq = map(np.min,filt_data['resp_delay_fits_Rsq'])
        deg = filt_data['resp_delay_deg']
        pref_dir = filt_data['PD']
        pd_S = map(str,list(map(int,pref_dir)))
        
        df_l = {}
        df_l['deg'] = deg
        df_l['mean_rsq'] = mean_rsq
        df_l['pd'] = pd_S
        df = pd.DataFrame.from_dict(df_l) 
        
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        ax=sns.jointplot(x=deg, y=mean_rsq, kind="kde", color=plt.cm.Dark2(3))
        
        ax.plot_joint(plt.scatter, c='w', s=10, linewidth=0.5, marker="o",
                       alpha=.0)
        sns.scatterplot('deg', 'mean_rsq', hue='pd',data=df,alpha=.8,s=30,
                        palette=[plt.cm.Dark2(4), plt.cm.Dark2(2)])
        ax.set_axis_labels(xlabel='Degrees ($^\circ$)',ylabel='$R^2$')
        
        if save_fig:
            # Saving figure 
            save_name = 'DelayProfiles_%s' % (exp_ID)
            os.chdir(fig_save_dir)
            fig1.savefig('%s.png'% save_name, bbox_inches='tight',
                        transparent=False,dpi=300)
            ax.savefig('Deg_vs_Rsq.png', bbox_inches='tight',
                        transparent=False,dpi=300)
            
    elif (analysis_type == 'stripes_ON_horRF_save_rois'):
        rois, all_rfs_sorted,max_epoch_traces  = \
            ROI_mod.generate_RF_profile_stripes(rois)
            
        ax = sns.heatmap(all_rfs_sorted)
        ax.set_xticklabels(rois[1].RF_profile_coords)
          
    elif ((analysis_type == 'stripes_ON_vertRF_transfer') or \
          (analysis_type == 'stripes_ON_horRF_transfer') or \
          (analysis_type == 'stripes_OFF_vertRF_transfer') or \
          (analysis_type == 'stripes_OFF_horRF_transfer')):
        
        rois = ROI_mod.generate_RF_map_stripes(rois, screen_w = 60)
        roi_traces = list(map(lambda roi: roi.df_trace, rois))
        roi_RF = list(map(lambda roi: roi.i_stripe_resp, rois))
        raw_stim = rois[0].stim_info['output_data']
        if (analysis_type == 'stripes_ON_vertRF_transfer'):
            figtitle = figtitle + '| ON_vert_RF'
        elif (analysis_type == 'stripes_ON_horRF_transfer'):
            figtitle = figtitle + '| ON_hor_RF'
        elif (analysis_type == 'stripes_OFF_vertRF_transfer'):
            figtitle = figtitle + '| OFF_vert_RF'
        elif (analysis_type == 'stripes_OFF_horRF_transfer'):
            figtitle = figtitle + '| OFF_hor_RF'
        for roi in rois:
            back_projected = np.tile(roi.i_stripe_resp, 
                                     (len(roi.i_stripe_resp),1) )
            if (analysis_type == 'stripes_ON_vertRF_transfer'):
                roi.vert_RF_ON_trace = roi.i_stripe_resp
                rotated = rotate(back_projected+1, angle=90)
                rotated= rotated-1
                roi.vert_RF_ON = rotated
                roi.vert_RF_ON_gauss = roi.stripe_gauss_profile
            elif (analysis_type == 'stripes_ON_horRF_transfer'):
                roi.hor_RF_ON_trace = roi.i_stripe_resp
                roi.hor_RF_ON = back_projected
                roi.hor_RF_ON_gauss = roi.stripe_gauss_profile
            elif (analysis_type == 'stripes_OFF_vertRF_transfer'):
                roi.vert_RF_OFF_trace = roi.i_stripe_resp
                rotated = rotate(back_projected+1, angle=90)
                rotated= rotated-1
                roi.vert_RF_OFF = rotated
                roi.vert_RF_OFF_gauss = roi.stripe_gauss_profile
            elif (analysis_type == 'stripes_OFF_horRF_transfer'):
                roi.hor_RF_OFF_trace = roi.i_stripe_resp
                roi.hor_RF_OFF = back_projected
                roi.hor_RF_OFF_gauss = roi.stripe_gauss_profile
        
        
        
        fig = sf.make_exp_summary_stripes(figtitle,analysis_params,
                                 rois[0].source_image,
                                 ROI_mod.get_masks_image(rois),
                                 roi_traces,raw_stim,roi_RF,save_fig,
                                 experiment_conditions['MovieID'],
                                 summary_save_dir)
                
        if save_fig:
            # Saving figure 
            save_name = 'RF_summary_%s' % (exp_ID)
            os.chdir(fig_save_dir)
            fig.savefig('%s.png'% save_name, bbox_inches='tight',
                        transparent=False)
    elif analysis_type == 'moving_gratings':
        map(lambda roi: roi.calculateTFtuning_BF(), rois)
        # Summary of current experiment
        data_to_extract = ['BF', 'SNR', 'reliability', 'uniq_id','exp_ID', 
                           'stim_name']
        roi_data = ROI_mod.data_to_list(rois, data_to_extract)
        roi_traces = list(map(lambda roi: roi.df_trace, rois))
        bf_image = ROI_mod.generate_colorMasks_properties(rois, 'BF')
        rois_df = pd.DataFrame.from_dict(roi_data)
        raw_stim = rois[0].stim_info['input_data']
        stim_info = rois[0].stim_info
        fig0=sf.make_exp_summary_TF(figtitle, 'aa',
                         rois[0].source_image,ROI_mod.get_masks_image(rois), 
                         roi_traces,raw_stim, bf_image,
                         rois_df, rois,stim_info, save_fig,
                         experiment_conditions['MovieID'],summary_save_dir)
        
        
        if save_fig:
           # Saving figure 
           f0_n = 'Summary_%s' % (exp_ID)
           os.chdir(fig_save_dir)
           fig0.savefig('%s.png'% f0_n, bbox_inches='tight',
                       transparent=False,dpi=300)
           if "RF_map" in rois[0].__dict__.keys():
               fig1 = ROI_mod.plot_RF_centers_on_screen(rois,prop='BF',
                                                 cmap='inferno',
                                                 ylab='BF (Hz)',
                                                 lims=(0,1.5))
               
               f1_n = 'BF_on_screen_%s' % (exp_ID)
               fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
                           transparent=False,dpi=300)
         
    
    elif analysis_type == 'gratings_transfer_rois_save':
        map(lambda roi: roi.calculateTFtuning_BF(), rois)
        # Summary of current experiment
        data_to_extract = ['DSI', 'BF', 'SNR', 'reliability', 'uniq_id',
                           'PD', 'exp_ID', 'stim_name']
        roi_data = ROI_mod.data_to_list(rois, data_to_extract)
        roi_traces = list(map(lambda roi: roi.df_trace, rois))
        bf_image = ROI_mod.generate_colorMasks_properties(rois, 'BF')
        rois_df = pd.DataFrame.from_dict(roi_data)
        raw_stim = rois[0].stim_info['input_data']
        stim_info = rois[0].stim_info
        fig0=sf.make_exp_summary_TF(figtitle, 'aa',
                         rois[0].source_image,ROI_mod.get_masks_image(rois), 
                         roi_traces,raw_stim, bf_image,
                         rois_df, rois,stim_info, save_fig,
                         experiment_conditions['MovieID'],summary_save_dir)
    
        fig1 = ROI_mod.plot_RFs(rois, number=20, f_w =5,cmap='coolwarm',
                                center_plot = True, center_val = 0.95)
        fig2 = ROI_mod.plot_RF_centers_on_screen(rois,prop='BF',
                                                 cmap='inferno',
                                                 ylab='BF (Hz)',
                                                 lims=(0,1.5))
        
        fig3 = ROI_mod.plot_RF_centers_on_screen(rois,prop='PD')
        
        if save_fig:
           # Saving figure 
           f0_n = 'Summary_%s' % (exp_ID)
           f1_n = 'RF_examples_%s' % (exp_ID)
           f2_n = 'BF_on_screen_%s' % (exp_ID)
           f3_n = 'PD_on_screen_%s' % (exp_ID)
           os.chdir(fig_save_dir)
           fig0.savefig('%s.png'% f0_n, bbox_inches='tight',
                       transparent=False,dpi=300)
           fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
                       transparent=False,dpi=300)
           fig2.savefig('%s.png'% f2_n, bbox_inches='tight',
                       transparent=False,dpi=300)
           
           fig3.savefig('%s.png'% f3_n, bbox_inches='tight',
                       transparent=False,dpi=300)
    
        for bf in np.unique(roi_data['BF']):
            curr_rois = np.array(rois)[np.array(roi_data['BF'])==bf]
            fig5 = ROI_mod.plot_RF_centers_on_screen(curr_rois,prop='BF',
                                       cmap='inferno',
                                       ylab='BF (Hz)',
                                       lims=(0,1.5))
            f5_n = 'BF_%.2f_onscreen' % (bf)
            os.chdir(fig_save_dir)
            fig5.savefig('%s.png'% f5_n, bbox_inches='tight',
                     transparent=False,dpi=300)
            
    elif analysis_type == 'STF_1':
        rois = ROI_mod.create_STF_maps(rois)
        # Summary of current experiment
        run_matplotlib_params()
        mean_STF = np.mean([np.array(roi.stf_map) for roi in rois],axis=0)
        fig = plt.figure(figsize = (5,5))
        plt.subplot(211)
        plt.title('STF map')
        ax=sns.heatmap(mean_STF, cmap='coolwarm',center=0,
                       xticklabels=np.array(rois[0].stf_map.columns.levels[1]).astype(int),
                       yticklabels=np.array(rois[0].stf_map.index),
                       cbar_kws={'label': '$\Delta F/F$'})
        ax.invert_yaxis()
        ax.invert_xaxis()
        
        plt.subplot(212)
        plt.title('STF map normalized')
        stf_map_norm=np.mean([np.array(roi.stf_map_norm) for roi in rois],axis=0)
        
        
        ax1=sns.heatmap(stf_map_norm, cmap='coolwarm',center=0,
                       xticklabels=np.array(rois[0].stf_map.columns.levels[1]).astype(int),
                       yticklabels=np.array(rois[0].stf_map.index),
                       cbar_kws={'label': 'zscore'})
        ax1.invert_yaxis()
        ax1.invert_xaxis()
        fig = plt.gcf()
        f0_n = 'Summary_STF_%s' % (exp_ID)
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f0_n, bbox_inches='tight',
                    transparent=False,dpi=300)
        
       
    return rois
            



def generate_pixel_maps(time_series,trialCoor,stimulus_information,frameRate,smooth=True,sigma=0.75):
    """
    """
    
    ## Generating pixel maps
    smooth_time_series = filters.gaussian(time_series, sigma=sigma)
    (wholeTraces_allTrials_smooth, respTraces_allTrials_smooth, baselineTraces_allTrials_smooth) = \
        separate_trials_video(smooth_time_series, trialCoor, stimulus_information,
                                                            frameRate)
    
    # Calculate maximum response
    MaxResp_matrix_all_epochs,  maxEpochIdx_matrix_all, \
    MeanResp_matrix_all_epochs, maxEpochIdx_matrix_all_mean = calculate_pixel_max(respTraces_allTrials_smooth,
                                                          stimulus_information)
    max_resp_matrix_all_mean = np.nanmax(MeanResp_matrix_all_epochs, axis=2)
    
    SNR_image = calculate_pixel_SNR(baselineTraces_allTrials_smooth,
                                    respTraces_allTrials_smooth,
                                    stimulus_information, frameRate,
                                    SNR_mode='Estimate')
    # DSI and CSI 
    DSI_image = create_DSI_image(stimulus_information, maxEpochIdx_matrix_all_mean,
                                                               max_resp_matrix_all_mean, MeanResp_matrix_all_epochs)
    mean_image = time_series.mean(0)
    CSI_image = create_CSI_image(stimulus_information,frameRate, 
                                 respTraces_allTrials_smooth, DSI_image)
    
    
    
    return SNR_image, DSI_image, CSI_image


    
    


def generate_roi_masks_image(roi_masks,im_shape):
    # Generating an image with all clusters
    all_rois_image = np.zeros(shape=im_shape)
    all_rois_image[:] = np.nan
    for index, roi in enumerate(roi_masks):
        curr_mask = roi
        all_rois_image[curr_mask] = index + 1
    return all_rois_image
    

def organize_extraction_params(extraction_type,
                               current_t_series=None,current_exp_ID=None,
                               alignedDataDir=None,
                               stimInputDir=None,
                               use_other_series_roiExtraction = None,
                               use_avg_data_for_roi_extract = None,
                               roiExtraction_tseries=None,
                               transfer_data_n = None,
                               transfer_data_store_dir = None,
                               transfer_type = None,
                               imaging_information=None,
                               experiment_conditions=None):
    
    extraction_params = {}
    extraction_params['type'] = extraction_type
    if extraction_type == 'SIMA-STICA':
        extraction_params['stim_input_path'] = stimInputDir
        if use_other_series_roiExtraction:
            series_used = roiExtraction_tseries
        else:
            series_used = current_t_series
        extraction_params['series_used'] = series_used
        extraction_params['series_path'] = \
            os.path.join(alignedDataDir, current_exp_ID, 
                                  series_used)
        extraction_params['area_max_micron'] = 4
        extraction_params['area_min_micron'] = 1
        extraction_params['cluster_max_1d_size_micron'] = 4
        extraction_params['cluster_min_1d_size_micron'] = 1
        extraction_params['extraction_reliability_threshold'] = 0.4
        extraction_params['use_trial_avg_video'] = \
            use_avg_data_for_roi_extract
    elif extraction_type == 'transfer':
        transfer_data_path = os.path.join(transfer_data_store_dir,
                                          transfer_data_n)
        extraction_params['transfer_data_path'] = transfer_data_path
        extraction_params['transfer_type']=transfer_type
        extraction_params['imaging_information']= imaging_information
        extraction_params['experiment_conditions'] = experiment_conditions
        
        
    return extraction_params

def refine_rois(rois, cat_bool, extraction_params,roi_1d_max_size_pixel,
                roi_1d_min_size_pixel,use_otsu=True, 
                mean_image=None,otsu_mask=None):
        
        if use_otsu:
            otsu_threshold_Value = filters.threshold_otsu(mean_image[otsu_mask])
            otsu_thresholded_mask = mean_image > otsu_threshold_Value
        else:
            otsu_thresholded_mask = cat_bool>-1
        
        [refined_rois, roi_image] = \
            clusters_restrict_size_regions(rois,cat_bool,roi_1d_max_size_pixel,
                 roi_1d_min_size_pixel,otsu_thresholded_mask)
        return refined_rois, roi_image
def select_properties_plot(rois , analysis_type):
    
    if analysis_type == 'gratings_transfer_rois_save' or\
        (analysis_type == 'STF_1'):
        
        
        properties = ['PD', 'DSI', 'CS','BF']
        colormaps = ['hsv', 'viridis', 'PRGn', 'inferno']
        
        if (analysis_type == 'STF_1'):
            vminmax = [(0,360), (0, 1), (-1, 1), (0, 1)]
        else:
            vminmax = [(0,360), (0, 2), (-1, 1), (0, 1.5)]
            
        
        data_to_extract = ['DSI', 'CSI', 'SNR', 'reliability','BF']
        
    elif ((analysis_type == '8D_10dps_stripes_RF') or\
         (analysis_type == '11_steps_luminance') or\
         (analysis_type == 'A_B_steps')   ):
        max_d = ROI_mod.data_to_list(rois, ['max_response'])
        max_snr = ROI_mod.data_to_list(rois, ['SNR'])
        properties = ['reliability', 'max_response' ,'SNR','reliability']
        colormaps = ['inferno', 'viridis','inferno','inferno']
        vminmax = [(0, 1), (0, np.max(max_d['max_response'])),
                   (0, np.max(max_snr['SNR'])),(0, 1)]
        data_to_extract = ['reliability', 'max_response' ,'SNR','reliability']
    
    elif ((analysis_type == 'luminance_edges_OFF' ) or\
          (analysis_type == 'luminance_edges_ON' )):
        
        properties = ['SNR', 'slope','reliability']
        colormaps = ['viridis', 'PRGn', 'viridis']
        vminmax = [(0, 3), (-1, 2), (0, 1)]
        data_to_extract = ['CSI', 'slope', 'reliability']
    
    elif (analysis_type == '5sFFF_analyze_save'):
        
        max_d = ROI_mod.data_to_list(rois, ['max_response'])
        properties = ['corr_fff', 'max_response']
        colormaps = ['PRGn', 'viridis']
        vminmax = [(-1,1), (0, np.max(max_d['max_response']))]
        data_to_extract = ['corr_fff', 'max_response']
        
    elif ((analysis_type == 'stripes_ON_vertRF_transfer') or \
          (analysis_type == 'stripes_ON_horRF_transfer') or \
          (analysis_type == 'stripes_OFF_vertRF_transfer') or \
          (analysis_type == 'stripes_OFF_horRF_transfer')):
        
        max_d = ROI_mod.data_to_list(rois, ['max_response'])
        max_snr = ROI_mod.data_to_list(rois, ['SNR'])
        properties = ['corr_fff', 'max_response' ,'SNR','reliability']
        colormaps = ['PRGn', 'viridis','inferno','inferno']
        vminmax = [(-1,1), (0, np.max(max_d['max_response'])),
                   (0, np.max(max_snr['SNR'])),(0, 1)]
        data_to_extract = ['stripe_gauss_fwhm', 'max_response' ,'SNR','reliability']
    
    elif ((analysis_type == 'stripes_ON_delay_profile') or \
          (analysis_type == 'stripes_OFF_delay_profile')):
        properties = ['PD', 'SNR', 'CS','reliability']
        colormaps = ['hsv', 'viridis', 'PRGn', 'viridis']
        vminmax = [(0,360), (0, 25), (-1, 1), (0, 1)]
        data_to_extract = ['DSI', 'CSI', 'resp_delay_deg', 'reliability']
        
    else:
        properties = ['SNR','reliability']
        colormaps = ['viridis','viridis']
        vminmax = [(0, 3),  (0, 1)]
        data_to_extract = ['SNR', 'reliability']
        
    return properties, colormaps, vminmax, data_to_extract

def run_ROI_selection(extraction_params, image_to_select=None):
    """

    """
    # Categories can be used to classify ROIs depending on their location
    # Backgroud mask (named "bg") will be used for background subtraction
    plt.close('all')
    plt.style.use("default")
    print('\n\nSelect categories and background')
    [cat_masks, cat_names] = select_regions(image_to_select, 
                                            image_cmap="gray",
                                            pause_t=8)
    
    # have to do different actions depending on the extraction type
    if extraction_params['type'] == 'manual':
        print('\n\nSelect ROIs')
        [roi_masks, roi_names] = select_regions(image_to_select, 
                                                image_cmap="gray",
                                                pause_t=4.5,
                                                ask_name=False)
        all_rois_image = generate_roi_masks_image(roi_masks,
                                                  np.shape(image_to_select))
        
        return cat_masks, cat_names, roi_masks, all_rois_image, None, None
            
    elif extraction_params['type'] == 'SIMA-STICA': 
        # Need the time series and information about the video to be extracted
        (time_series, stimulus_information,imaging_information) = \
            pre_processing_movie (extraction_params['series_path'],
                                  extraction_params['stim_input_path'])
        
        # A trial averaged version of the video can be used for extraction 
        # since it may decrease the noise. Yet it can introduce artifacts
        # between the epochs
        if extraction_params['use_trial_avg_video']:
            (avg_video, _, _) = \
                separate_trials_video(time_series,stimulus_information,
                                      imaging_information['frame_rate'])
            sima_dataset = generate_avg_movie(extraction_params['series_path'], 
                                              stimulus_information,
                                              avg_video)
        else:
            movie = np.zeros(shape=(time_series.shape[0],1,time_series.shape[1],
                                time_series.shape[2],1))
            movie[:,0,:,:,0] = time_series
            b = sima.Sequence.create('ndarray',movie)
            sima_dataset = sima.ImagingDataset([b],None)
        
        # We need a certain range of areas for rois 
        area_max_micron = extraction_params['area_max_micron']
        area_min_micron = extraction_params['area_min_micron']
        area_max = int(math.pow(math.sqrt(area_max_micron) / \
                                imaging_information['pixel_size'], 2))
        area_min = int(math.pow(math.sqrt(area_min_micron) / \
                                imaging_information['pixel_size'], 2))
        [roi_masks, all_rois_image] = find_clusters_STICA(sima_dataset,
                                                          area_min,
                                                          area_max)
        threshold_dict = {'SNR': 0.75,'reliability': 0.4}
        
        return cat_masks, cat_names, roi_masks, all_rois_image, None, threshold_dict
    
    elif extraction_params['type'] == 'transfer':
        
        rois = run_roi_transfer(extraction_params['transfer_data_path'],
                                extraction_params['transfer_type'],
                                experiment_info=extraction_params['experiment_conditions'],
                                imaging_info=extraction_params['imaging_information'])
        
        return cat_masks, cat_names, None, None, rois, None
    
    else:
       raise TypeError('ROI selection type not understood.') 
    
    
    
    
    
    
def interpolate_data_dyuzak(stimtimes, stimframes100hz, dsignal, imagetimes, freq):
    """Interpolates the stimulus frame numbers (*stimframes100hz*), signal
    traces (*dsignal*) by using the
    stimulus time (*stimtimes*)  and the image time stamps (*imagetimes*)
    recorded. Interpolation is done to a frequency (*freq*) defined by the
    user.
    recorded in

    Parameters
    ----------
    stimtimes : 1D array
        Stimulus time stamps obtained from stimulus_output file (with the
        rate of ~100Hz)
    stimframes100hz : 1D array
        Stimulus frame numbers through recording (with the rate of ~100Hz)
    dsignal : mxn 2D array
        Fluorescence responses of each ROI. Axis m is the number of ROIs while
        n is the time points of microscope recording with lower rate (10-15Hz)
    imagetimes : 1D array
        The time stamps of the image frames with the microscope recording rate
    freq : int
        The desired frequency to interpolate

    Returns
    -------
    newstimtimes : 1D array
        Stimulus time stamps with the rate of *freq*
    dsignal : mxn 2D array
        Fluorescence responses of each ROI with the rate of *freq*
    imagetimes : 1D array
        The time stamps of the image frames with the rate of *freq*
    """
    
    # Interpolation of stimulus frames and responses to freq

    # Creating time vectors of original 100Hz (or 60Hz) (x)  and freq Hz sampled(xi)
    # x = vector with 100Hz (or 60Hz) rate, xi = vector with user input rate (freq)
    x = np.linspace(0,len(stimtimes),len(stimtimes))
    xi = np.linspace(0,len(stimtimes),
                     np.round(int((np.max(stimtimes)-np.min(stimtimes))*freq)+1))

    # Get interpolated stimulus times for 20Hz
    # stimtimes and x has same rate (100Hz (or 60Hz))
    # and newstimtimes is interpolated output of xi vector
    newstimtimes = np.interp(xi, x, stimtimes)
    newstimtimes =  np.array(newstimtimes,dtype='float32')

    # Get interpolated stimulus frame numbers for 20Hz
    # Below stimframes is a continuous function with stimtimes as x and
    # stimframes100Hz as y values
    stimframes = interpolate.interp1d(stimtimes,stimframes100hz,kind='nearest')
    # Below interpolated stimulus times are given as x values to the stimtimes
    # function to find interpolated stimulus frames (y value)
    stimframes = stimframes(newstimtimes)
    stimframes = stimframes.astype('int')

    #Adjusting the imagetimes (from microscope) length to the total number of frames in stim_output file 
    # (Microscope starts recording timing before the stimulus is presented)
    # (Therefore the first times at the beggining need to be removed)
    remove_times = len(imagetimes) - len(dsignal)
    imagetimes = imagetimes[remove_times:]                       
                        

    #Get interpolated responses for 20Hz
    dsignal1 = np.empty(shape=(len(dsignal),
                               len(newstimtimes)),dtype=dsignal.dtype)
    dsignal=np.interp(newstimtimes, imagetimes, dsignal)


    return (newstimtimes, dsignal, stimframes)
    
    
    

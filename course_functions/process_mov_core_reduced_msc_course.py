#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:42:35 2020

@author: burakgur
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
try:
    import cPickle # For Python 2.X
except:
    import pickle as cPickle # For Python 3.X
import random
import warnings
from numpy.fft import fft, fftfreq

from skimage import io
from scipy import interpolate
from scipy.stats.stats import pearsonr
from itertools import permutations
from roipoly import RoiPoly, MultiRoi
from skimage import filters
from scipy.ndimage.interpolation import rotate

import ROI_mod_reduced_msc_course as ROI_mod
#import summary_figures as sf
from xmlUtilities_course import getFramePeriod, getLayerPosition, getPixelSize,getMicRelativeTime
from core_functions_reduced_msc_course import readStimOut, readStimInformation, getEpochCount, divide_all_epochs
from core_functions_reduced_msc_course import divideEpochs
from post_analysis_core import run_matplotlib_params

#%%
def pre_processing_movie (dataDir,stimInputDir,stack):
    

    # Generate necessary directories for figures
    current_t_series=os.path.basename(dataDir)
    
    # Load movie, get stimulus and imaging information
    try: 
        # movie_path = os.path.join(dataDir, 'Mot_corr_frames.tif') # Seb: changed by the line below
        movie_path = os.path.join(dataDir, stack)
        time_series = io.imread(movie_path)
    except IOError:
        movie_path = os.path.join(dataDir, '{t_name}_{stack}'.format(t_name=current_t_series, stack=stack))
        time_series = io.imread(movie_path)
        
    ## Get stimulus and xml information
    (stimulus_information, imaging_information) = \
        get_stim_xml_params(dataDir, stimInputDir)
    
    
    
    return time_series, stimulus_information,imaging_information
#%%
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
                                          skipHeader=3) # Seb: skipHeader = 3 for _stimulus_ouput from 2pstim
    
    # Stimulus information
    (stimInputFile,stimInputData) = readStimInformation(stimType=stimType,
                                                      stimInputDir=stimInputDir)
    isRandom = int(stimInputData['randomize'][0])
    epochDur = stimInputData['duration']
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
     
    
    # Transfering all data from input file to stimulus_information
    
    stimulus_data = stimInputData
    stimulus_information = stimulus_data.copy()


    # Adding more information
    stimulus_information['epoch_dur'] = epochDur # Seb: consider to delete this line. Redundancy
    stimulus_information['random'] = isRandom # Seb: consider to delete this line. Redundancy
    stimulus_information['output_data'] = rawStimData 

    stimulus_information['frame_timings'] = imagetimes
    stimulus_information['input_data'] = stimInputData # Seb: consider to delete this line. Redundancy
    stimulus_information['stim_name'] = stimType.split('/')[-1]
    stimulus_information['trial_coordinates'] = trialCoor

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


    # Keeping imaging information
    imaging_information = {'frame_rate' : frameRate, 'pixel_size': x_size, 
                             'depth' : depth}
        
    return stimulus_information, imaging_information
#%%
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
    #JC: removed SIMA option
    extraction_params = {}
    extraction_params['type'] = extraction_type
    if extraction_type == 'transfer':
        transfer_data_path = os.path.join(transfer_data_store_dir,
                                          transfer_data_n)
        extraction_params['transfer_data_path'] = transfer_data_path
        extraction_params['transfer_type']=transfer_type
        extraction_params['imaging_information']= imaging_information
        extraction_params['experiment_conditions'] = experiment_conditions
        
        
    return extraction_params

#%% Functions for ROI selection

def run_ROI_selection(extraction_params, stack, image_to_select=None):
    """

    """
    # Categories can be used to classify ROIs depending on their location
    # Backgroud mask (named "bg") will be used for background subtraction
    plt.close('all')
    plt.style.use("default")
    print('\n\nSelect categories and background')
    [cat_masks, cat_names] = select_regions(image_to_select, 
                                            image_cmap="viridis",
                                            pause_t=8)
    
    # have to do different actions depending on the extraction type
    #JC: removed SIMA option
    if extraction_params['type'] == 'manual':
        print('\n\nSelect ROIs')
        [roi_masks, roi_names] = select_regions(image_to_select, 
                                                image_cmap="viridis",
                                                pause_t=4.5,
                                                ask_name=False)
        all_rois_image = generate_roi_masks_image(roi_masks,
                                                  np.shape(image_to_select))
        
        return cat_masks, cat_names, roi_masks, all_rois_image, None, None
    
    elif extraction_params['type'] == 'transfer':
        
        rois = run_roi_transfer(extraction_params['transfer_data_path'],
                                extraction_params['transfer_type'],
                                experiment_info=extraction_params['experiment_conditions'],
                                imaging_info=extraction_params['imaging_information'])
        
        return cat_masks, cat_names, None, None, rois, None
    
    else:
       raise TypeError('ROI selection type not understood.') 


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
        plt.imshow(mask_agg, alpha=0.3,cmap = 'tab20b')
        plt.title("Select ROI: ROI%d" % roi_number)
        plt.show(block=False)
       
        
        # Draw ROI
        curr_roi = RoiPoly(color='r', fig=fig)
        iROI = iROI + 1
        # plt.waitforbuttonpress()
        # plt.pause(pause_t)
        if ask_name:
            try:
                mask_name = raw_input("\nEnter the ROI name:\n>> ") # Python 2.X
            except:
                mask_name = input("\nEnter the ROI name:\n>> ") # Python 3.X
            
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
        try:
            signal = raw_input("\nPress k for exiting program, otherwise press enter") # Python 2.X
        except:
            signal = input("\nPress k for exiting program, otherwise press enter") # Python 3.X
        
        if (signal == 'k\r'):
            stopsignal = 1
        elif (signal == 'k\\r'):
            stopsignal = 1
        elif (signal == 'k'):
            stopsignal = 1
        
    
    return roi_masks, mask_names


def generate_roi_masks_image(roi_masks,im_shape):
    # Generating an image with all clusters
    all_rois_image = np.zeros(shape=im_shape)
    all_rois_image[:] = np.nan
    for index, roi in enumerate(roi_masks):
        curr_mask = roi
        all_rois_image[curr_mask] = index + 1
    return all_rois_image


def run_roi_transfer(transfer_data_path, transfer_type,experiment_info=None,
                     imaging_info=None):
    '''
    Updates:
        25/03/2020 - Removed transfer types of 11 steps and AB steps since they
        are redundant with minimal type
    '''
    load_path = open(transfer_data_path, 'rb')
    workspace = cPickle.load(load_path)
    rois = workspace['final_rois']
    
    if transfer_type == 'lumgratings'  or \
        transfer_type == 'lum_con_gratings' :
        
        properties = ['CSI', 'CS','PD','DSI','category',
                      'analysis_params']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    
    else:
        raise NameError('Invalid ROI transfer type')
        
        
   
    return transferred_rois

#%% seperate trials

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
        current_epoch_dur = stimulus_information['duration'][iEpoch]
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

#%% Signal to Noise ratio and Correlation between first and last trial

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

#%% Plotting ROIs and properties
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
    col_row_n = int(math.ceil(math.sqrt(total_n_images))) #Seb: added 'int'
    
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

#%% Analysis dependend on stimulus type, here only lumgrating!
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
    
    if analysis_type == 'lumgratings'  or analysis_type == 'noisygratings' or analysis_type == 'TFgratings':
        
        rois = ROI_mod.analyze_gratings_general(rois)
        run_matplotlib_params()
        #mean_TFL = np.mean([np.array(roi.tfl_map) for roi in rois],axis=0)
        mean_TFL = np.mean([np.array(roi.tfl_map).astype(float) for roi in rois],axis=0) #JC: for Python3, need to get values in float
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

        # Plotting ROI traces per epoch for visualitation
        for i,roi in enumerate(rois):
            epochs_roi_data = roi.whole_trace_all_epochs
            if roi.stim_info['stimtype'][1] == 'lumgrating': # Seb: based on the second epoch information
                variable_name = 'lum'
            elif roi.stim_info['stimtype'][1] == 'noisygrating':
                variable_name = 'SNR'

            variable_of_interest = roi.stim_info[variable_name]

            max_value = -100 #Seb: a extremely negative value to start with
            min_value = 100 #Seb: a extremely positive value to start with
            for epoch in epochs_roi_data:
                curr_max = max(epochs_roi_data[epoch])
                curr_min = min(epochs_roi_data[epoch])
                if curr_max > max_value:
                    max_value = curr_max
                if curr_min < min_value:
                    min_value = curr_min

            # Constructing the plot backbone
            figtitle = 'ROI #:{} \n {} \n {}'.format(i,roi.stim_name,roi.experiment_info['Genotype']) 
            fig_Rois = plt.figure(figsize=(8.3, 11.7)) # A4 size in inches
            fig_Rois.suptitle(figtitle,fontsize=12)
            coln = 4
            rown = 4
            grid = plt.GridSpec(rown, coln, wspace=0.4, hspace=0.3)

            #Plotting whole trace
            ax1=plt.subplot(grid[0,0:4])
            ax1.plot(roi.df_trace,lw=1,color='k')
            ax1.set_xticks([])            
            ax1.set_ylim((min_value-0.5),(max_value+0.5))
            ax1.set_ylabel('$\Delta F/F$')
            ax1.set_title('Whole trace')

            #Plotting every epoch trace after trial averaging
            cur_row = 0
            for idx, epoch in enumerate(epochs_roi_data):
                        
                if np.mod(idx,coln) == 0:
                    cur_row +=1
                curr_trace = epochs_roi_data[epoch]
                        
                ax2=plt.subplot(grid[cur_row,np.mod(idx,coln)])
                ax2.plot(curr_trace,lw=2,color='k')
                ax2.legend()
                ax2.set_xticks([])
                ax2.set(frame_on=False)                 
                ax2.set_ylim((min_value-0.5),(max_value+0.5))
                ax2.set_ylabel('$\Delta F/F$')
                ax2.set_title('%.3f' % variable_of_interest[epoch])  

            #Plotting every epoch 1hz amplitude of the fft spectrum
            #Third to last subplot in th grid for general info
            ax3=plt.subplot(grid[3,1])
            ax3.plot(variable_of_interest[1:],roi.fft_X_hz_amp, 'ko')
            ax3.set_ylabel('Amplitude (arb units)')
            ax3.set_xlabel(variable_name)
            ax3.set_title('1hz power')

            #Plotting every epoch NORMALIZED 1hz amplitude of the fft spectrum.
            #Second to last subplot in th grid for general info
            norm_fft_1hz_amp = [x / roi.fft_X_hz_amp[0] for x in roi.fft_X_hz_amp]
            ax4=plt.subplot(grid[3,2])
            ax4.plot(variable_of_interest[1:],norm_fft_1hz_amp, 'ko')
            ax4.set_ylim(0,1)
            ax4.set_xlabel(variable_name)
            ax4.set_title('Normalized 1hz power') #JC: why do we not see the other datapoints?
            
            #Last subplot in the grid for general info
            ax5=plt.subplot(grid[3,3])
            ax5.set_xlim(0,1)
            ax5.set_ylim(0,1)
            y_start = 0.5
            inter_text = 0.1 
            ax5.text(0.1,(y_start+inter_text),'Reliability: {0:.3f}'.format(roi.reliability),dict(size=8))            
            inter_text += 0.1
            ax5.text(0.1,(y_start+inter_text) ,'Age: {}'.format(roi.experiment_info['Age']),dict(size=8))
            inter_text += 0.1
            ax5.text(0.1,(y_start+inter_text) ,'Sex: {}'.format(roi.experiment_info['Sex']),dict(size=8))
            inter_text += 0.1
            ax5.text(0.1,(y_start+inter_text) ,'Epochs variable: {}'.format(variable_name),dict(size=8))
            ax5.axis('off')

            os.chdir(fig_save_dir)
            fig_Rois.savefig('ROI_%d.png'% i, bbox_inches='tight',
                    transparent=False,dpi=300)
    return rois


#%%
def select_properties_plot(rois , analysis_type):
    
    if analysis_type == 'lumgratings':
        properties = ['SNR','reliability']
        colormaps = ['viridis','viridis']
        vminmax = [(0, 3),  (0, 1)]
        data_to_extract = ['SNR', 'reliability']
        
    return properties, colormaps, vminmax, data_to_extract

#%%
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
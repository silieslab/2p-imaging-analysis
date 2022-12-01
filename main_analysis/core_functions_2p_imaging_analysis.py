# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:44:39 2021

@authors: Burak Gur, Juan F. Vargas, Jacqueline Cornean, Sebastian Molina-Obando
"""
#%% Importing packages

import os
import glob
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats.stats import pearsonr
from scipy.interpolate import NearestNDInterpolator
from itertools import permutations
import math

from main_analysis.xml_functions import getFramePeriod,getMicRelativeTime,getLayerPosition,getPixelSize
from main_analysis.stim_functions import readStimOut, readStimInformation
from main_analysis.epoch_functions import getEpochCount, divideEpochs, divide_all_epochs
from post_analysis.post_analysis_core import run_matplotlib_params




#%% Functions
def load_movie (dataDir,stack):
    

    # Generate necessary directories for figures
    current_t_series=os.path.basename(dataDir)
    
    # Load movie, get stimulus and imaging information
    movie_path = os.path.join(dataDir, stack)  # stack meaning tif stack of motion alinged images-JC
    time_series = io.imread(movie_path)        # this does the loading part to get the movie/ tif stack -JC


    return time_series



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
        
    imaging_information : XXXXXXXXX
    
    stimType : XXXXXXXXX 
    rawStimData : XXXXXXXXX
    stimInputFile : XXXXXXXXX
        


    """
    
    #%% Metadata from microscope computer xml file
    
    # Finding the xml file and retrieving relevant information
    
    xmlPath = os.path.join(t_series_path, '*-???.xml')  # '*and ?' are wild card characters, maches 0/ more or 1 character respectivly -JC
    xmlFile = (glob.glob(xmlPath))[0]   # glob returns file path matching a specific pattern -JC
    
    #  Finding the frame period (1/FPS) and layer position
    framePeriod = getFramePeriod(xmlFile=xmlFile)
    frameRate = 1/framePeriod
    layerPosition = getLayerPosition(xmlFile=xmlFile)
    depth = layerPosition[2]        # get Z position from layerPostion
    
    imagetimes = getMicRelativeTime(xmlFile)

    
    # Pixel definitions
    x_size, y_size, pixelArea = getPixelSize(xmlFile)
    
    # Keeping imaging information
    # creating a dictionary to store the infos from the metadata -JC
    imaging_information = {'frame_rate' : frameRate, 'pixel_size': x_size, 
                             'depth' : depth, 'frame_timings' : imagetimes}
    
    
   #%% Metadata from stimulus computer input and output files
    
    # Stimulus output information
    stimOutPath = os.path.join(t_series_path, '*_stimulus_output_*')
    stimOutFile = (glob.glob(stimOutPath))[0]   # if there are more stimulus_outputfiles it will take the first one -> [0] -JC
    (stimType, rawStimData) = readStimOut(stimOutFile=stimOutFile, 
                                          skipHeader=3) # Seb: skipHeader = 3 for _stimulus_ouput from 2pstim
        # get path of the stim output file (stimtyp) and an array with the stim output data -JC
    
    # Stimulus information
    (stimInputFile,stimulus_information) = readStimInformation(stimType=stimType,
                                                      stimInputDir=stimInputDir)
    #stimInputFile: path to stim input file, stim_info: dict of parameters from stim input
    #file -JC

    return imaging_information, stimulus_information, stimType, rawStimData,stimInputFile
    
    
    
    
def get_epochs_identity(imaging_information,stimulus_information,stimType, rawStimData,stimInputFile):
    """ Gets specific info about each epoch 
    Parameters
    ==========
   
        
    Returns
    =======
 
    stimulus_information : containing 
                            trialCoor : list of start, end coordinates for each trial for each epoch
                            XXXXXXXXX :
                                ...
    """
    
    isRandom = int(stimulus_information['randomize'][0])    #get first value of 'randomize' -JC
    epochDur = stimulus_information['duration']
    epochDur = [float(sec) for sec in epochDur]     #transform in float (but aren't they already?) -JC
    epochCount = getEpochCount(rawStimData=rawStimData, epochColumn=3)
    framePeriod = 1/imaging_information['frame_rate']
    # Finding epoch coordinates and number of trials, if isRandom is 1 then
    # there is a baseline epoch otherwise there is no baseline epoch even 
    # if isRandom = 2 (which randomizes all epochs)                                        
    if epochCount <= 1:
        trialCoor = 0
        trialCount = 0
    elif isRandom == 1:
        #don't get this function? -JC
        (trialCoor, trialCount, _) = divideEpochs(rawStimData=rawStimData,
                                                 epochCount=epochCount,
                                                 isRandom=isRandom,
                                                 framePeriod=framePeriod ,
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
     

    # Adding more information
    stimulus_information['epoch_dur'] = epochDur # Seb: consider to delete this line. Redundancy
    stimulus_information['random'] = isRandom # Seb: consider to delete this line. Redundancy
    stimulus_information['output_data'] = rawStimData
    stimulus_information['frame_timings'] = imaging_information['frame_timings']
    stimulus_information['stim_name'] = stimType.split('/')[-1]
    stimulus_information['trial_coordinates'] = trialCoor

    if isRandom==0: #why is here a baseline epoch? -JC
        stimulus_information['baseline_epoch'] = 0 #-JC None instead of 0
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
    

        
    return stimulus_information

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
    
    """ THIS IS DOING THIS
    Parameters
    ==========
   
    XXXXXXXXXXX
        
    Returns
    =======
 
    XXXXXXXXXXX  
    
    
    """
    
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
        extraction_params['area_max_micron'] = 4        #how to get these valuse? -JC
        extraction_params['area_min_micron'] = 1        #only for T4 and T5 axon terminals -JC
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
            perm = permutations(range(trial_numbers), 2)    #get all data pairs possible in the array with shape of trial_numbers -JC
            coeff =[]
            for iPerm, pair in enumerate(perm):
                curr_coeff, pval = pearsonr(currentRespTrace[:-2,pair[0]],
                                            currentRespTrace[:-2,pair[1]])
                coeff.append(curr_coeff)        #calculate pearson correlation for all possible traces -JC
                
            coeff = np.array(coeff).mean()
            
            noise_std = currentBaseTrace.std(axis=0).mean(axis=0)   #std of mean? -JC
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
    fig_dim1 = 5
    fig_dim2 = 5*(roi_image.shape[0]/roi_image.shape[1]) # fig dimention based on original roi_image shape
    fig1, ax1 = plt.subplots(ncols=1, nrows=1,facecolor='k', edgecolor='w',
                             figsize=(fig_dim1,  fig_dim2))
    
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
        #plt.savefig('%s.pdf'% save_name, bbox_inches='tight')
        print('ROI images saved')


def plot_all_trials(respTraces_allTrials_ROIs,Tseries_folder,save_fig = False):
    """
    
    Meant for plotting when ROI number > 1
    """

    for e, rois in enumerate(respTraces_allTrials_ROIs.items()):
        #Rois is a tuple containing [epoch, dict_of rois] 
        num_subplots = len(rois[1])
        fig,axes = plt.subplots(ncols=num_subplots, nrows=1,facecolor='k', edgecolor='w',
                            figsize=(25, 5))
        fig.suptitle(f'Epoch: {e}') # or fig.subtitle(f'Epoch: {rois[0]}')
            
        for i,ax in enumerate(axes): 
            roi = rois[1][i]
            ax.plot(roi)
            ax.set_title(f'ROI #{i}')
            ax.set_ylabel('dF/F')
            ax.set_xlabel('recording frames')
        # Saving figure
        if save_fig:
            trial_folder = os.path.join(Tseries_folder,'Trials')
            if not os.path.exists(trial_folder):
                os.mkdir(trial_folder)
            save_name = f'\\Epoch_{e}' 
                
            plt.savefig(f'{trial_folder+save_name}.png', bbox_inches='tight')
            plt.close()
            print(f'ROI trials saved here: {trial_folder}')

def plot_conc_trace(rois,exp_ID,Tseries_folder,save_fig = False):
    """
    Meant for plotting when ROI number > 1
    """
    num_subplots = len(rois)
    fig,axes = plt.subplots(ncols=num_subplots, nrows=1,facecolor='k', edgecolor='w',
                        figsize=(50, 7))
    fig.suptitle(f'Trial average concatenated response') 

    for r, ax in enumerate(axes):
        time_trace = np.linspace(0, len(roi.int_conc_trace)/roi.int_rate, num =len(roi.int_conc_trace))
        roi = rois[r]
        color = 'tab:blue'
        ax.plot(time_trace,roi.int_conc_trace,label='response',color=color)
        ax.set_title(f'ROI #{r}')
        ax.set_ylabel('dF/F', color=color)  
        ax.set_xlabel('time (s)')
        ax.tick_params(axis='y', labelcolor=color)
        ax.legend(loc=(0, 0.95))

        ax2 = ax.twinx()  # initiate a second axes that shares the same x-axis
        color = 'tab:orange'
        ax2.set_ylabel('epochs', color=color)  
        ax2.plot(time_trace,roi.int_stim_trace,label='stimulus',color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc=(0, 0.9))
        
    # Saving figure
    if save_fig:
        trial_average_folder = Tseries_folder 
        if not os.path.exists(trial_average_folder):
            os.mkdir(trial_average_folder)
        save_name = '\\ROI_trial_average_conc_trace_%s' % (exp_ID)      
        plt.savefig(f'{trial_average_folder+save_name}.png', bbox_inches='tight')
        plt.savefig(f'{trial_average_folder+save_name}.pdf', bbox_inches='tight')
        plt.close()
        print(f'ROI trial average saved here: {trial_average_folder}')

def select_properties_plot(rois , analysis_type):

    if analysis_type == 'general':
        properties = ['SNR','reliability']
        colormaps = ['viridis','viridis']
        vminmax = [(0, 3),  (0, 1)]
        data_to_extract = ['SNR', 'reliability']

    return properties, colormaps, vminmax, data_to_extract

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
        #plt.savefig('%s.pdf'% save_name, bbox_inches='tight',dpi=300)
        print(f'ROI properties saved here: {save_dir}')

def interpolate_signal(signal, sampling_rate, int_rate, trace_type, stim_duration = 10):
    """
    """
     #juan: corrected interpolation
    period=1/sampling_rate  #JC: sec/frame
    timeV=  np.linspace(period,(len(signal)+1)*period,num=len(signal))
    # Create an interpolated time vector in the desired interpolation rate
    #JC: x-coordinates of data points
    stim_duration = int(stim_duration)
    steps = stim_duration*int_rate #how many steps for the time vector depends on int_rate and stimulus duration/ epoch duration -JC
    #JC: changed hard coded timeVI
    #Juan suggestion of replacing 0.1 with 1/int_rate
    timeVI=np.linspace(1/int_rate,stim_duration, steps) # Seb: this was right. 
    #timeVI=np.linspace(1/int_rate,stim_duration*int_rate, steps)# Seb: this was woring

    #Interpolation
    interp_traces = np.interp(timeVI, timeV, signal)

    #TODO Ask Jacqueline why the if statement below, for now, it is commented out
    if trace_type == 'stim':
        for index, value in enumerate (interp_traces):
            if value > 0 and value < 1: #select the value in between 0 and 1 in the stim traces, which was interpolated and replace it with 0 (no slope!) -JC
                interp_traces[index] = 0
    return interp_traces
def conc_traces(rois, interpolation = True, int_rate = 10):
    """
    Concatanates and interpolates traces.
    
    """
    for roi in rois:
        conc_trace = []
        stim_trace = []
        prev_stim_dur = 0 #JC
        #TODO Solve the issue of somtime having the epoch starting from 0 ot 1
        #It should be related to stimulus_information['random']
        start = roi.stim_info['EPOCHS']-len(roi.whole_trace_all_epochs.keys())
        try:
            for idx, epoch in enumerate(list(range(start,roi.stim_info['EPOCHS']))): #Seb: epochs_number --> EPOCHS
                #if not roi.stim_info['texture.duration'] and (roi.stim_info['duration'][idx]==0):  #JC: for noise stim
                curr_stim_dur = roi.stim_info['duration'][epoch]  #This is wrong. The interpolated trace is sometimes a combination of epochs
                curr_stim_dur = round(len(roi.whole_trace_all_epochs[epoch])/roi.imaging_info['frame_rate']) # This is rigth
                #elif roi.stim_info['texture.duration'] and (roi.stim_info['duration'][idx]==0):
                #    curr_stim_dur = roi.stim_info['texture.duration'][idx] * roi.stim_info['texture.count'][idx] #for noise
                    
                stimulus_dur = curr_stim_dur + prev_stim_dur    #JC update stimulus duration with concatination
                prev_stim_dur = stimulus_dur                    #JC
                curr_stim = np.zeros((1,len(roi.whole_trace_all_epochs[epoch])))[0]
                curr_stim = curr_stim + epoch
                stim_trace=np.append(stim_trace,curr_stim,axis=0)
                conc_trace=np.append(conc_trace,roi.whole_trace_all_epochs[epoch],axis=0)
                #getting epoch 0 and 1 together? but still same size?   changed 1 to 0
        except:
            for idx, epoch in enumerate(list(range(0,roi.stim_info['EPOCHS']))): #Seb: epochs_number --> EPOCHS JC: changed start to 0 to include first epoch
                #if not roi.stim_info['texture.duration'] and (roi.stim_info['duration'][idx]==0):  #JC: for noise stim
                curr_stim_dur = roi.stim_info['duration'][idx]  #JC
                #elif roi.stim_info['texture.duration'] and (roi.stim_info['duration'][idx]==0):
                #    curr_stim_dur = roi.stim_info['texture.duration'][idx] * roi.stim_info['texture.count'][idx] #for noise
                    
                
                stimulus_dur = curr_stim_dur + prev_stim_dur    #JC update stimulus duration with concatenation
                prev_stim_dur = stimulus_dur                    #JC
                curr_stim = np.zeros((1,len(roi.whole_trace_all_epochs[epoch])))[0]
                curr_stim = curr_stim + idx
                stim_trace=np.append(stim_trace,curr_stim,axis=0)
                conc_trace=np.append(conc_trace,roi.whole_trace_all_epochs[epoch],axis=0)
                #getting epoch 0 and 1 together? but still same size?   changed 1 to 0

        scaled_stim_trace = stim_trace/(max(stim_trace)/max(conc_trace))
        roi.conc_trace = conc_trace
        roi.stim_trace = stim_trace
        roi.conc_stim_duration = stimulus_dur
        roi.scaled_stim_trace =scaled_stim_trace 
        
        # Calculating correlation
        curr_coeff, pval = pearsonr(roi.conc_trace,roi.stim_trace)
        roi.corr_fff = curr_coeff
        roi.corr_pval = pval
        if interpolation:
            roi.int_conc_trace = interpolate_signal(conc_trace, 
                                                   roi.imaging_info['frame_rate'], 
                                                   int_rate, 'data', stimulus_dur)
            roi.int_stim_trace = interpolate_signal(stim_trace, 
                                                    roi.imaging_info['frame_rate'], 
                                                   int_rate, 'stim', stimulus_dur)
            roi.int_rate = int_rate

            roi.int_scaled_stim_trace = roi.int_stim_trace/(max(roi.int_stim_trace)/max(roi.int_conc_trace))
            
    return rois
# %%

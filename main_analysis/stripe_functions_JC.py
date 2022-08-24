# env 2pImaging python 3.9.7
# -*- coding: utf-8 -*-

'''
Created on 22.03.2022

@author: Jacqueline Cornean (JC)

Functions needed to analyze standing stripes stimulus
'''
#%%
#%% Importing packages
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from cProfile import label
import seaborn as sns
import pandas as pd
import numpy as np
import operator
from scipy.stats import pearsonr

os.chdir(r'\\fs02\jcornean$\Dokumente\PhD\python_github\2p-imaging-analysis-branch_develop_main')
from roi_class_JC import ROI_bg, movingaverage
import ROI_mod


#%%get epoch identity for stripe stimulation

#code peace for standing stripe analysis, getting epochs and trials:
#need to find a way to get only the start of the nan, right now it is every value
def divide_epochs_stripes(rawStimData, frame_period, trial_diff=0.20,check_last_trial_len=True):
    '''
    Function to find all the start and end values for all epochs and all trials
    
    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    frame_period : float
        Time it takes to image a single frame.

    trial_diff : float
        Default: 0.20
        A safety measure to prevent last trial of an epoch being shorter than
        the intended trial duration, which can arise if the number of frames
        was miscalculated for the t-series while imaging. *Effective if and
        only if check_last_trial_len is True*. The value is used in this way
        (see the corresponding line in the code):

        *(len_first_trial - len_last_trial) * frame_period >= trial_diff*

        If the case above is satisfied, then this last trial is not taken into
        account for further calculations.
        Edited code peace from *divide_all_epochs* function.

    check_last_trial_len: Boulean
        If True check if last trial has the same length as the rest

    Returns
    =======

    epoch_dict: dict
        dict in dict, first one has the epoch number as key. The value of this key is another dict
        with values start: start of the different trials, value is the corresponding microscopy frame (data)
                    end:   end of the whole epoch (with nan lines), value same a start
                    x_pos: x coordinates of the bar position on the screen
                    x_nan_pos: start of when the x_pos corrdinates are nan
                                --> stim is gray and no bar is shown, meaning it can be used as a bg period
                                --> can be used as end of bar stimulation
                                --> value is the corresponding microscopy frame (data)
    '''
    epoch_dict = {}
    epoch_number = np.unique(rawStimData[:,2])
    epoch_number = [int(number) for number in epoch_number]
    for epoch in epoch_number:
        start_list = []
        end_list = []
        x_pos_list = []
        all_nan_start = []
        curr_nans = []
        previouse_line = 0
        pre_nan_line = 0
        for index, line in enumerate(rawStimData):
            if line[2]== epoch: #column with boutInd --> used to idendify epochs
                curr_line = line[7] #column with microscope frames
                if curr_line-previouse_line>1:
                    start_list.append(curr_line)
                    if not previouse_line == 0:
                        end_list.append(previouse_line)
                previouse_line = curr_line

                if line[4]-pre_nan_line>0 or line[4]-pre_nan_line<0:
                    x_pos_list.append(line[4]) #x coordinates of stripe
                    pre_nan_line = line[4]
                
                if math.isnan(line[4]) == True: #get list of nans for baseline/ f0
                    curr_nans.append(curr_line)
                else:
                    if len(curr_nans) >1: #ignore single nan lines at the beginning of epoch
                        all_nan_start.append(curr_nans[0])
                        curr_nans = []
                    else:
                        curr_nans = []
        end_list.append(previouse_line)
        if len(curr_nans) >1:
            all_nan_start.append(curr_nans[0])
                                                
        epoch_dict[epoch] = {'start':start_list, 'end':end_list, 'x_pos':x_pos_list, 'x_nan_pos':all_nan_start}

    if check_last_trial_len == True:
        for epoch in epoch_dict:
            len_first_trial = epoch_dict[epoch]['end'][0]-epoch_dict[epoch]['start'][0]
            len_last_trial = epoch_dict[epoch]['end'][-1]-epoch_dict[epoch]['start'][-1]

            if (len_first_trial-len_last_trial)*frame_period >= trial_diff:
                print(f'Last trial of epoch {epoch} is discarded since it was too short')
                epoch_dict[epoch]['start'].pop(-1)
                epoch_dict[epoch]['end'].pop(-1)
    
    return epoch_dict, len(epoch_number)


#%% JC_2022_04_25: get epochs for strip stimulation

def get_epochs_for_stripes(imaging_information,stimulus_information,stimType, rawStimData):
    '''
    Function to save different variables in stimulus_information, e.g
    epoch_dict with the start and stop values of all epochs.
    
    Patameters:
    ============
    imaging_informaion: dict
        Dict with all the imaging info inside, e.g frame rate, imaging time...

    stimulus_information: dict
        Dict with all the stimulus informations stored in.
        One key of them is epoch_dict, which is a dict, containing start, end and x_pos
        of each epoch and trial.
    
    stimType: string
        Name of the used stimulus, which is also the name of the stim_input file.

    rawStimData: array

    '''
    #code peace for standing stripe analysis, getting epochs and trials:
    
    epoch_duration = stimulus_information['bar.duration']
    bg_duration = stimulus_information['bg.duration']
    frame_period = 1/imaging_information['frame_rate']

    epoch_dict, epoch_number = divide_epochs_stripes(rawStimData, frame_period)

    # Adding the info to stimulus_information dictionary
    stimulus_information['epoch_dur'] = epoch_duration
    stimulus_information['baseline_duration'] = bg_duration
    stimulus_information['epoch_adjuster'] = 0
    stimulus_information['epoch_dict'] = epoch_dict
    stimulus_information['epoch_count'] = epoch_number
    stimulus_information['output_data'] = rawStimData
    stimulus_information['frame_timings'] = imaging_information['frame_timings']
    stimulus_information['frame_period'] = frame_period
    stimulus_information['stim_name'] = stimType.split('/')[-1]
    
    return stimulus_information

#%%
#trying to get df/f with the inter_stim_interval
def get_df_inter_stim_intervall (time_series, rois, stimulus_information, moving_avg = True, bins = 3 ):
    '''
    Function to calculate df/f by using the last half of the inter stimulus intervall
    prior to each epoch (here with the variable curr_baseline). The mean of it is f0.
    The values will be stored in the roi. class.

    Parameters
    ===========
    time_series: ndarray
        Microscope data read in from the motion corrected TIFF stack.

    rois: class
        Object with informations of each ROI stored inside.

    stimulus_information: dict
        Dict with all the stimulus informations stored in.
        One key of them is epoch_dict, which is a dict, containing start, end and x_pos
        of each epoch and trial.
    
    moving_avg: bool
        True or False, if moving averaging should be done
    
    bins: int
        Bin size for the moving_averaging fuction. Default is 3.
    '''

    frame_rate = rois[0].imaging_info['frame_rate']
    number_f_zero_frames = int(frame_rate*0.5) #get number of frames we need to for half (0.5sec) of the inter_stim_intervall (1sec)
    epoch_one = stimulus_information['epoch_dict'][1] #need for last epoch to get end frame
    for iROI, roi in enumerate(rois):
        roi.raw_trace = time_series[:,roi.mask].mean(axis=1) #get values for raw trace
        #include first 5 secounds without stimulus in df/F
        df_f_inter_stim = (roi.raw_trace[0:int(epoch_one['start'][0])]-roi.raw_trace[0:int(epoch_one['start'][0])].mean())/roi.raw_trace[0:int(epoch_one['start'][0])].mean()
        for trial in range(len(epoch_one['start'])): #number of trials
            try:
                for epoch in stimulus_information['epoch_dict']:
                    curr_epoch = stimulus_information['epoch_dict'][epoch]
                    curr_start = int(curr_epoch['start'][trial]) #start of the epoch, frame of the microscope
                    curr_baseline = roi.raw_trace[int(curr_epoch['start'][trial]-number_f_zero_frames): curr_start]
                    f_zero = curr_baseline.mean()
                    
                    if epoch == 18:     #end is start of the next epoch 1
                        df_inter_stim_per_epoch = (roi.raw_trace[curr_start:int(epoch_one['start'][trial+1])] - f_zero)/f_zero
                    else:
                        next_epoch = stimulus_information['epoch_dict'][epoch+1]
                        try:
                            df_inter_stim_per_epoch = (roi.raw_trace[curr_start:int(next_epoch['start'][trial])] - f_zero)/f_zero
                        except IndexError:
                            df_inter_stim_per_epoch = (roi.raw_trace[curr_start:int(curr_epoch['end'][trial])] - f_zero)/f_zero
                    df_f_inter_stim = np.append(df_f_inter_stim, df_inter_stim_per_epoch) #append df/f single traces to one big trace

            except IndexError:
                continue
        if moving_avg:
            roi.df_trace = movingaverage(df_f_inter_stim, bins) #do moving average
        else:
            roi.df_trace = df_f_inter_stim
    
# %% trial averaging

def trial_averaging_stripes (rois, stimulus_information, df_true = True):

    wholeTraces_allTrials_ROIs = {}
    respTraces_allTrials_ROIs = {}
    respTraces_allTrials_ROIs_raw = {}
    baselineTraces_allTrials_ROIs = {}

    frame_rate = rois[0].imaging_info['frame_rate']
    
    for epoch in stimulus_information['epoch_dict']:

        wholeTraces_allTrials_ROIs[epoch] = {}
        respTraces_allTrials_ROIs[epoch] = {}
        respTraces_allTrials_ROIs_raw[epoch] = {}
        baselineTraces_allTrials_ROIs[epoch] = {}

        whole_lens = []
        resp_lens = []
        base_lens = []

        curr_epoch = stimulus_information['epoch_dict'][epoch]
        if epoch == 18:
            continue

        next_epoch = stimulus_information['epoch_dict'][epoch+1]
        number_f_zero_frames = int(frame_rate*0.5)
        trial_number = len(curr_epoch['start'])
        for trial in range(trial_number):
            curr_start, curr_end, pre_baseline_start, stim_end = get_trial_coord(trial, curr_epoch,
                                                                                next_epoch,
                                                                                number_f_zero_frames)

            whole_trace_len = curr_end - pre_baseline_start
            whole_lens.append(whole_trace_len)
            resp_trace_len = stim_end-curr_start
            resp_lens.append(resp_trace_len)
            base_trace_len = curr_start-pre_baseline_start
            base_lens.append(base_trace_len)
        
        trial_len =  min(whole_lens)
        resp_len = min(resp_lens)
        base_len = min(base_lens)

        for index_roi, roi in enumerate (rois):
            #generating shape of array, which is needed to average the trials later
            wholeTraces_allTrials_ROIs[epoch][index_roi] = np.zeros(shape=(trial_len,
                                                                            trial_number))
            respTraces_allTrials_ROIs[epoch][index_roi] = np.zeros(shape=(resp_len,
                                                                            trial_number))
            respTraces_allTrials_ROIs_raw[epoch][index_roi] = np.zeros(shape=(resp_len,
                                                                            trial_number))
            baselineTraces_allTrials_ROIs[epoch][index_roi] = np.zeros(shape=(base_len,
                                                                            trial_number))

            for trial in range(trial_number):
                curr_start, curr_end, pre_baseline_start, stim_end = get_trial_coord(trial, curr_epoch,
                                                                                             next_epoch,
                                                                                             number_f_zero_frames)
                curr_trial_whole_trace = roi.df_trace[pre_baseline_start:curr_end]
                wholeTraces_allTrials_ROIs[epoch][index_roi][:,trial] = curr_trial_whole_trace[:trial_len]

                curr_trial_resp_traces = roi.df_trace[curr_start:stim_end]
                respTraces_allTrials_ROIs[epoch][index_roi][:,trial] = curr_trial_resp_traces[:resp_len]

                curr_trial_resp_traces_raw = roi.raw_trace[curr_start:stim_end]
                respTraces_allTrials_ROIs_raw[epoch][index_roi][:,trial] = curr_trial_resp_traces_raw[:resp_len]

                if df_true == True:
                    curr_base_trace = roi.df_trace[pre_baseline_start:curr_start]
                else:
                    curr_base_trace = roi.raw_trace[pre_baseline_start:curr_start]
                baselineTraces_allTrials_ROIs[epoch][index_roi][:,trial] = curr_base_trace[:base_len]

            
            wt = np.nanmean(wholeTraces_allTrials_ROIs[epoch][index_roi],axis=1)
            rt = np.nanmean(respTraces_allTrials_ROIs[epoch][index_roi],axis=1)
            rt_raw = np.nanmean(respTraces_allTrials_ROIs_raw[epoch][index_roi],axis=1)
            bt = np.nanmean(baselineTraces_allTrials_ROIs[epoch][index_roi],axis=1)

            roi.appendTrace(wt,epoch, trace_type = 'whole')
            roi.appendTrace(rt,epoch, trace_type = 'response')
            roi.appendTrace(rt_raw,epoch, trace_type = 'response_raw')

    return (wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs,respTraces_allTrials_ROIs_raw, 
            baselineTraces_allTrials_ROIs)

# %%

def get_trial_coord(trial, curr_epoch, next_epoch, number_f_zero_frames):

    try:
        curr_end = int(next_epoch['start'][trial]-number_f_zero_frames)
    except IndexError:
        curr_end = int(curr_epoch['end'][trial]-number_f_zero_frames)
    curr_start = int(curr_epoch['start'][trial])
    pre_baseline_start = int(curr_epoch['start'][trial]-number_f_zero_frames)
    stim_end = int(curr_epoch['end'][trial]-(number_f_zero_frames*2))

    return curr_start, curr_end, pre_baseline_start, stim_end
#%%plot single epoch averaged
def plot_averaged_trials (int_whole_traces, roi_id, save_path):
    plt.rcdefaults()
    fig, axs = plt.subplots(len(int_whole_traces),1, figsize=(6,8), sharey=True)
    for epoch in int_whole_traces:
        e_index = epoch-1
        curr_int_trace = int_whole_traces[epoch]
        zero_line = np.zeros(curr_int_trace.shape)
        axs[e_index].plot(curr_int_trace)
        axs[e_index].plot(zero_line, 'gray')
        axs[e_index].axis('off')
    axs[-1].axis('on')
    axs[-1].spines['top'].set_visible(False)
    axs[-1].spines['right'].set_visible(False)
    axs[-1].set_xlabel('frames', fontsize=14)
    axs[-1].set_ylabel('df/f', fontsize=10)
    fig.text(0.04, 0.5, 'Stripe positions', va='center',
                         ha='center', rotation='vertical', fontsize=14)
    saved_plot_n = os.path.join(save_path,'trial_average')
    if not os.path.exists(saved_plot_n):
        os.mkdir(saved_plot_n)
    saved_plot_name = os.path.join(saved_plot_n, f'averaged_trials_{roi_id}')
    fig.savefig(saved_plot_name)



#%% FUNKTIONS FOR POST-ANALYSIS
def plot_SNR_density (rois, saved_plots_dir, save_option):
    '''
    Plot the density distribution of the SNR and reliability values
    of ROIs.
    Parameters
    ============
    rois: list of ROIs 

    save_option: Boolean (True or False)
            Option if the plot should be saved or not

    Returns
    ============
    generates a plot
     '''
    #get dict of SNR and reliability
    dict_rois = ROI_mod.data_to_list(rois, ['SNR', 'reliability'])

    #plot density of them
    plt.rcdefaults()
    fig, axs = plt.subplots(1,2)
    sns.distplot(ax = axs[0], x = dict_rois['SNR'],rug=True, hist = False)
    sns.distplot(ax = axs[1], x = dict_rois['reliability'],rug=True, hist = False)
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axs[0].set_xlabel('SNR')
    axs[1].set_xlabel('reliability')

    saved_plot_name = os.path.join(saved_plots_dir, '%s_SNR_reliability'%(rois[0].experiment_info['MovieID']))
    if save_option == True:
        fig.savefig(saved_plot_name)
# %% 
def thresholding (rois,threshold=0.7, SNR_threshold=0.8):
    '''Threshold over the reliability and SNR of each ROI.
    Parameters:
    ============
    rois: list of ROIs
    
    threshold: integer
                value for the reliability threshold. Every ROI below it
                will not be used.
    
    SNR_threshold: integer
                value for the SNR threshold. Every ROI below it
                will not be used.

    Returns
    ============
    used_rois: list
                list of all the ROIs that were over both thresholds.
    '''
    used_rois = []
    for roi in rois:
        if roi.reliability< threshold or roi.SNR< SNR_threshold:
            continue
        else:
            used_rois.append(roi)
    return used_rois
#%% get sorted epochs
def get_sorted_epochs (roi):
    '''
    Function to extract the epochs with their respective x_postition
    and sort the epochs depending on the x_pos starting from the lowest value.

    Parameters
    ============
    roi: single roi for witch the sorting should be done

    Returns
    ============
    sorted_epochs_dict: dict
                    dict containing sorted epoch numbers and
                    the respective x_position of the stripe on the screen.
                    Keys: epoch and x_pos
    
    sorted_epoch_numbers: list
                    list of sorted epoch IDs
    '''
    x_pos_dict = {}
    sorted_epochs_dict = {}
    epoch_dict = roi.stim_info['epoch_dict']
    for epoch in epoch_dict:
        if epoch_dict[epoch]['x_pos'] == []:
            continue
        else:
            x_pos_dict[epoch] = epoch_dict[epoch]['x_pos']

    sorted_epochs = sorted(x_pos_dict.items(),key=operator.itemgetter(1))
    sorted_epoch_numbers = [sorted_epochs[index][0] for index in range(len(sorted_epochs))]
    
    for i in range(len(sorted_epochs)): #get it back into a dict
        sorted_epochs_dict[i] = {'epoch':sorted_epochs[i][0], 'x_pos':sorted_epochs[i][1]}
    
    return sorted_epochs_dict, sorted_epoch_numbers

#%%
def get_max_value (sorted_trace_single, sorted_epochs_dict, epoch_num):
    '''Get maximum value and index of each epoch.
    Parameters:
    ============
    sorted_trace_single: array
                array of all values of the trace of the current epoch.
    
    sorted_epochs_dict: dict
                dict containing sorted epoch numbers and
                the respective x_position of the stripe on the screen.
                Keys: epoch and x_pos

    epoch_num: integer
                index of the current epoch number. 

    Returns:
    ============
    sorted_epochs_dict: dict
                updated dict which includes the maximum values and 
                indeces of each epoch.
    '''
    epoch_max = np.nanmax(sorted_trace_single)
    epoch_max_idx = np.nanargmax(sorted_trace_single)
    sorted_epochs_dict[epoch_num]['epoch_max'] = epoch_max
    sorted_epochs_dict[epoch_num]['epoch_max_idx'] = epoch_max_idx

    return sorted_epochs_dict

def get_sorted_traces (roi, sorted_epoch_numbers, sorted_epochs_dict):
    '''
    Function to get the traces of the sorted epochs and concatinate them.
    
    Parameters:
    ============
    roi: single roi for witch the sorting should be done

    sorted_epoch_numbers: sorted_epoch_numbers: list
                    list of sorted epoch IDs
    
    sorted_epochs_dict: dict
                    dict containing sorted epoch numbers and
                    the respective x_position of the stripe on the screen.
                    Keys: epoch and x_pos
    
    Returns:
    ============
    sorted_traces: array
                    array of the sorted traces concatinated.
                    The trace pices of each epoch are sorted according to the
                    x_position of the epochs.
    '''

    sorted_traces = roi.int_whole_trace_all_epochs[sorted_epoch_numbers[0]]
    for index, epoch_num in enumerate (sorted_epoch_numbers):
        sorted_trace_single = roi.int_whole_trace_all_epochs[epoch_num]
        sorted_epochs_dict = get_max_value (sorted_trace_single,
                        sorted_epochs_dict, epoch_num = index )
        if epoch_num == sorted_epoch_numbers[0]:
            continue
        else:
            sorted_traces = np.append(sorted_traces, sorted_trace_single)
    return sorted_traces, sorted_epochs_dict

#%%
def plot_sorted_traces(used_rois, saved_plots_dir, save_option, date):
    '''Generating a plot of the sorted traces of all ROIs
    
    Parameters:
    ============
    used_rois: list of all the ROIs that were better than the reliability
            and the SNR thresholds.
    
    saved_plots_dir: directory of where to save the plot

    save_option: Boolean (True or False)
            Option if the plot should be saved or not

    Returns:
    ============
    used_rois: updated version

    generates a plot

    '''
    plt.rcdefaults()
    fig, axs = plt.subplots(len(used_rois),1, figsize=(8,5), sharey=True)
    for index, roi in enumerate(used_rois):        
        sorted_epochs_dict, sorted_epoch_numbers = get_sorted_epochs(roi)
        sorted_traces, sorted_epochs_dict = get_sorted_traces(roi, 
                            sorted_epoch_numbers, sorted_epochs_dict)
        roi.sorted_traces = sorted_traces
        roi.sorted_epochs_dict = sorted_epochs_dict
        roi.sorted_epoch_numbers = sorted_epoch_numbers
        zero_line = np.zeros(sorted_traces.shape)
        axs[index].plot(sorted_traces)
        axs[index].plot(zero_line, 'gray')
        axs[index].axis('off')
    axs[-1].axis('on')
    axs[-1].spines['top'].set_visible(False)
    axs[-1].spines['right'].set_visible(False)
    axs[-1].set_xlabel('frames', fontsize=14)
    axs[-1].set_ylabel('df/f', fontsize=10)
    fig.text(0.04, 0.5, 'ROIs', va='center', ha='center', rotation='vertical', fontsize=14)
    saved_plot_name = os.path.join(saved_plots_dir, f'{date}_sorted_traces')
    if save_option == True:
        fig.savefig(saved_plot_name)
    else:
        print ('Sorted traces were not saved.')
    
    return used_rois

#%% 
def centering_singel_ROI (traces):
    '''Center each trace of every ROI, so that it's peak is in the center of the plot.
    Like this, it is better to compare different neurons with each other.
    
    Parameters:
    ============
    traces: array
            the current trace of the ROI which one want to center. 
            Can be e.g the whole trace or just the maxima of each epoch.
            
    Returns:
    ============
    centered_trace: array
            The centered input trace.
    '''
    max_value = np.nanmax(traces)
    max_index = np.nanargmax(traces)
    if max_index < int(len(traces)/2):
        trace_index_to_center = int(len(traces)/2)+max_index
    else:
        trace_index_to_center = np.absolute(int(len(traces)/2)-max_index)
    centered_trace = np.append(traces[trace_index_to_center:],
                                traces[:trace_index_to_center])
    return centered_trace

#%% center the sorted traces
def centering_traces (used_rois, saved_plots_dir, date, save_option):
    '''Center each trace of every ROI, so that it's peak is in the center of the plot.
    Like this, it is better to compare different neurons with each other.
    
    Parameters:
    ============
    used_rois: list of all the ROIs that were better than the reliability
            and the SNR thresholds.
            
    Returns:
    ============
    centered_list: list of the centered traces.
    '''
    centered_list = []
    plt.rcdefaults()
    fig, axs = plt.subplots(len(used_rois),1, figsize=(8,5), sharey=True)
    for index, roi in enumerate(used_rois):
        sorted_traces = roi.sorted_traces
        centered_trace = centering_singel_ROI(sorted_traces)
        roi.centered_trace = centered_trace
        centered_list.append(centered_trace)
        zero_line = np.zeros(sorted_traces.shape)
        axs[index].plot(centered_trace)
        axs[index].plot(zero_line, 'gray')
        axs[index].axis('off')

    axs[-1].axis('on')
    axs[-1].spines['top'].set_visible(False)
    axs[-1].spines['right'].set_visible(False)
    axs[-1].set_xlabel('frames', fontsize=14)
    axs[-1].set_ylabel('df/f', fontsize=10)
    fig.text(0.04, 0.5, 'ROIs', va='center', ha='center', rotation='vertical', fontsize=14)
    saved_plot_name = os.path.join(saved_plots_dir, f'{date}_centered_traces')
    if save_option == True:
        fig.savefig(saved_plot_name)
    return (centered_list)

#%%plot heatmap of multiple rois
def plot_centered_traces_and_heatmap (used_rois, saved_plots_dir, date, save_option):
    '''
    Plot the centered traces of all rois and 
    a corresponding heatmap (strength of df/F)
    
    Parameters:
    ============
    used_rois: list of all the ROIs that were better than the reliability
            and the SNR thresholds.
            
    Returns:
    ============
    heatmap and plot of traces
    '''
    plt.rcdefaults()
    centered_list = centering_traces(used_rois, saved_plots_dir, date, save_option)
    colour = sns.diverging_palette(145, 300, s=60, as_cmap=True)
    plt.figure()
    sns.heatmap(centered_list, vmin = -1.5, vmax = 1.5, xticklabels=False, cmap = colour)
    saved_plot_name = os.path.join(saved_plots_dir, f'{date}_heatmap_centered_traces')
    if save_option == True:
        plt.savefig(saved_plot_name)


#%% get the gaussian fit

def make_gaussian_fit (used_rois):
    #make linespace for the stim/epochs to be able to interpolate the data
    screen_begin = used_rois[0].sorted_epochs_dict[0]['x_pos'] #get the coord of one side of the screen
    screen_end = used_rois[0].sorted_epochs_dict[16]['x_pos'] #get the coord of the other side
    screen_w = int(np.absolute(screen_begin)+np.absolute(screen_end))
    screen_coords = np.linspace(0, screen_w, num=screen_w,
                                                endpoint=True)
    for index, roi in enumerate(used_rois):
        roi_max = np.linspace(0, screen_w, num=len(roi.sorted_epochs_dict),
                                                        endpoint=True)
        epoch_max = [roi.sorted_epochs_dict[i]['epoch_max'] 
                        for i in range(len(roi.sorted_epochs_dict))]
        centered_max = centering_singel_ROI(epoch_max)
        
        max_int = np.interp(screen_coords, roi_max, epoch_max)
        fit_trace, r_squared, coeff = ROI_mod.fit_1d_gauss(screen_coords, max_int)
        roi.gauss_fit_trace = fit_trace
        roi.stripe_gauss_coeff = coeff #[A=np.max(data_y), mu=np.argmax(data_y), sigma]
        roi.stripe_gauss_fwhm = 2.355 * coeff[2] #FWHM=2sqrt(2ln2)sigma = 2.3548sigma 
        roi.stripe_r_squared = r_squared

        centered_max_int = np.interp(screen_coords, roi_max, centered_max)
        centered_fit_trace, r_squared_centered,\
                         coeff_centered = ROI_mod.fit_1d_gauss(screen_coords, 
                         centered_max_int)
        roi.centered_gauss_fit_trace = centered_fit_trace
        roi.centered_gauss_coeff = coeff_centered #[A=np.max(data_y), mu=np.argmax(data_y), sigma]
        roi.centered_gauss_fwhm = 2.355 * coeff_centered[2] #FWHM=2sqrt(2ln2)sigma = 2.3548sigma 
        roi.centered_r_squared = r_squared_centered

    return used_rois

#%% list of fwhm values and max amplitude
def plot_fwhm_max_resp (used_rois, saved_plots_dir, date, save_option):
    fwhm = []
    fwhm_c = []
    max_response = []
    plt.rcdefaults()
    for roi in used_rois:
        fwhm.append(roi.stripe_gauss_fwhm)
        fwhm_c.append(roi.centered_gauss_fwhm)
        max_response.append(roi.max_response)
    response_df = pd.DataFrame(data={'fwhm':fwhm, 'centered_fwhm':fwhm_c,
                                    'max_response':max_response})
    fig, axs = plt.subplots(1,2, figsize=(8,5), sharex=True)
    sns.violinplot(y = 'fwhm', data=response_df, ax=axs[0])
    sns.swarmplot(y = 'fwhm', data=response_df, color='k', ax=axs[0])
    #max response
    sns.violinplot(y = 'max_response', data=response_df, ax=axs[1])
    sns.swarmplot(y = 'max_response', data=response_df, color='k', ax=axs[1])
    saved_plot_name = os.path.join(saved_plots_dir, f'{date}_FWHM_max_resp')
    if save_option == True:
        plt.savefig(saved_plot_name)

#just fitted traces
def plot_gauss_fit_trace (used_rois,saved_plots_dir, date, save_option,centered=True):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    for roi in used_rois:
        if centered ==True:
            trace_type = roi.centered_gauss_fit_trace
        else:
            trace_type = roi.gauss_fit_trace
        plt.plot(trace_type)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Screen position')
    ax.set_ylabel('df/f')
    saved_plot_name = os.path.join(saved_plots_dir, f'{date}_gauss_fit')
    if save_option == True:
        plt.savefig(saved_plot_name)
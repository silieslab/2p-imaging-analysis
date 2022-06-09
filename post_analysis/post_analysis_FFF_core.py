# env 2pImaging python 3.9.7
# -*- coding: utf-8 -*-

'''
Created on 22.03.2022

@author: Jacqueline Cornean (JC)

Code for Full Field Flash Analysis data, to plot first responses of the neurons.
Here are the core functions to plot the graphs
'''
#%% Import packages
from cProfile import label
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
os.chdir(r'U:\Dokumente\python_github\2p-imaging-analysis-branch_develop_main')
from roi_class import ROI_bg

#%% sort negatively and positivly correlated cells
#pearson correlation is better than spearman and enought to say of ON or OFF
def cell_correlation (fly_data): 
    pos_corr = []
    neg_corr = []
    for roi_index, roi in enumerate(fly_data):
        pe_corr = roi.corr_fff
        #print ('before interpolation: ', roi.corr_fff)
        #pe_corr, pe_pval = pearsonr(roi.int_con_trace, roi.int_stim_trace)
        #print ('after interpolation: ', pe_corr)
        if pe_corr > 0:
            pos_corr.append(roi)
        else:
            neg_corr.append(roi)
        
    return (pos_corr, neg_corr)

#%% thresholding through reliability
def threshold_reliabilty (fly_data, threshold):
    if threshold is None:
        print('No threshold used.')
        return fly_data
    
    used_rois = []
    for roi in fly_data['rois']:
        if roi.reliability > threshold:
            used_rois.append(roi)
        
    return used_rois

#%% load pickle file
def load_data_function (dataname, threshold, x_mode='all'):
    '''
    Parameters
    ============
    dataname: path and name of the pickle file in which the data is stored

    x_mode: decide if you want to plot all single ROIs or only positiv or negative correlated ROIs
            choose between: all, positive, negative 
    
    Returns
    ============
    fly_data: dict with keys 'rois' and 'roi_image'. All the data is stored in the
            values of the key 'rois', which includes objects of the class ROI_bg.'''

    fly_data = pickle.load(open(dataname, 'rb'))

    #thresholding ROIs:
    thresholded_data = threshold_reliabilty(fly_data, threshold)

    x_mode = x_mode
    pos_corr, neg_corr = cell_correlation(thresholded_data)
    if x_mode == 'all':
        data = thresholded_data['rois']
        title = 'All ROIs'
    elif x_mode == 'positive':
        data = pos_corr
        title = 'Positive correlated ROIs'
    elif x_mode == 'negative':
        data = neg_corr
        title = 'Negative correlated ROIs'
    else:
        print ('Warning: the x_mode is not correct! Please choose between all, positive or negative.')


    return data, title

#%% function to hold defold parameters for plots
def plot_params (title = None, x_label = 'time in seconds', y_label = 'df/f'):
    '''Here are the defold parameters of the plot backbone, like x and y labels, grit visibility...
    
    Parameters
    =============
    title: titel of the plot, will be made in the load_data function while choosing the
    x_mode (to plot all, negative correlated or positive correlated traces

    x_label: titel of the x axis, by default 'time in seconds'

    y_label: titel of the y axis, by default 'df/f'
    
    Returns
    ============
    Basic plot structure'''

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(15,5), gridspec_kw={'height_ratios': [1,8]})
    fig.suptitle(title, fontsize=22) #setting plot parameters
    axs[1].set_xlabel(x_label, fontsize=18)
    axs[1].set_ylabel(y_label, fontsize=18)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[0].axis('off')
    axs[1].tick_params(axis='both', labelsize=15)

    return (fig, axs)
#%%

def get_time_vector_x(roi):
    '''Generating the time_vector_x, which is an array holding values corresponding
    to the stimulus/ recording time and will be used as x coordinates in plots.
    
    Parameters:
    ============
    roi: current roi in fly data
    
    Returns:
    time_vector_x: np.array, eqaually spaced with step size: interpolation rate * duration
    
    time_half: length of half the time, used generate the stimulus trace and to solve
                the problem with the apperent slope between epochs, when there should be a strict change.
                
    epoch0: np.array of zeros for 0 lum epoch in FFF
                
    epoch1: np.array of ones for 100% lum epoch in FFF
            time_half+1 to start the epoch where the oder ended'''

    steps = int(roi.int_rate*roi.conc_stim_duration)
    time_vector_x = np.linspace(1/roi.int_rate,roi.conc_stim_duration, steps)
    time_half = int(len(time_vector_x)/2)
    epoch0 = np.zeros(time_half,) #for stim plot to not have a slope
    epoch1 = np.ones(time_half+1,)

    return (time_vector_x, time_half, epoch0, epoch1)

# %% plotting single ROIs of one fly
def plot_single_ROIs (pickle_list, threshold, x_mode, saved_plots_dir, save=False):
    '''Plot traces of each single ROI of one fly with the stimulus profile above
    
    Parameters
    ============
    pickle_list: list of path and name of pickle files from one genotype and stim type
               to load the data
    
    x_mode: decide if you want to plott all single ROIs or only positiv or negative correlated ROIs
            choose between: all, positive, negative
               
    Returns
    ===========
    For each fly, it returns and saves a plot of single ROI traces.
    The path it will be saved in, is defined in saved_plots_dir, listed above '''

    for index, file in enumerate(pickle_list):
        data, title = load_data_function(file, threshold, x_mode)
        if len(data) == 0:
            continue

        fig, axs = plot_params (title)
        roi_id_list = []
        for roi_index, roi in enumerate(data): 
            concat_int_trace = roi.int_con_trace
            stim_trace = roi.int_stim_trace
            roi_id = roi.uniq_id
            
            roi_id_list.append(roi_id)

            time_vector_x, time_half, epoch0, epoch1 = get_time_vector_x(roi)

            # axs[1].plot(time_vector_x, concat_int_trace)
            axs[0].plot(time_vector_x[0:time_half], epoch0 , 'black')
            axs[0].plot(time_vector_x[time_half-1:], epoch1 , 'black')
            axs[0].vlines(time_vector_x[time_half-1], 0, 1 , 'black')
            sns.lineplot(x=time_vector_x, y=concat_int_trace, ax=axs[1], ci=95)
            plt.legend(labels=roi_id_list)

        if save == True:
            saved_plot_name = os.path.join(saved_plots_dir, '%s_single_ROIs_one_fly_%s'%(roi.experiment_info['MovieID'], x_mode))
            fig.savefig(saved_plot_name)

    
#%% averaging all ROIs for of one fly
def plot_mean_roi_multiple_flies (pickle_list, threshold, x_mode, saved_plots_dir, date, save=False):
    '''Plot the mean of all ROIs of one fly in one single trace,
    plot traces of several flies in one plot
    
    Parameters
    ============
    pickle_list: list of path and name of pickle files from one genotype and stim type
               to load the data
    
    Returns
    ===========
    Plot of mean trace of a fly for multiple flies, which are saved in the directory
    named saved_plots_dir
    '''
    #setting plotting parameters
    #title will be placed below because we need x_mode and here
    #we load the data after setting the plotting params --> several flies in 1 plot
    #and x_mode, which influences the titel is in the load_data_function
    fig, axs = plot_params ()
    fly_ID_list = []
    #load data for each fly and averaging ROIs
    for index, file in enumerate(pickle_list):
        fly_data, title = load_data_function(file, threshold, x_mode)

        if len(fly_data) == 0:
            continue

        #create time vector for x coordinates (same for every roi)
        roi0 = fly_data[0]
        time_vector_x, time_half, epoch0, epoch1 = get_time_vector_x(roi0)

        #get flyID for legend
        fly_ID = roi0.experiment_info['FlyID']
        fly_ID_list.append(fly_ID)

        #create a list with all rois of one fly
        roi_traces_list = []
        #mean_traces = []
        all_traces = pd.DataFrame(columns=['frame', 'datapoint', 'timepoint']) #create a df with info you need to plot data with sns later
        
        for roi in fly_data:
            roi_traces_list.append(roi.int_con_trace)
        #average rois over each frame of one fly
        for frame in range (len(roi_traces_list[0])):
            for roi_index, roi in enumerate(roi_traces_list):
                curr_traces = pd.DataFrame(columns=['frame', 'datapoint', 'timepoint']) #temporal df for each roi
                curr_frame = roi[frame]
                curr_traces.loc[frame, 'frame'] = frame #add corresponding frame to df
                curr_traces.loc[frame, 'datapoint'] = curr_frame    #add df/f value for this frame and ROI
                curr_traces.loc[frame, 'timepoint'] = time_vector_x[frame]  #add timepoint which would correspond to the frame number (important for plotting x values)
                all_traces = all_traces.append(curr_traces, ignore_index=True)  #append the temporal df in a global one to save all the data
            
                                                                                #ignore_index = True important to not get same index values, if same index --> does not work
            #the next 4 lines are helpful if you want to plot only with plt and not with seaborn. Here I manually calculate the mean of all ROIs per frame
            #     curr_list = np.append(curr_list, curr_frame)    #create list with same frame of each Roi
            # mean_frame = np.mean(curr_list) #mean of all ROIs for the same frame
            # mean_traces = np.append(mean_traces, mean_frame)    #array of averaged trace of one fly
        #all_traces.append(mean_traces)
        
        
        #plotting traces
        axs[0].plot(time_vector_x[0:time_half], epoch0 , 'black')
        axs[0].plot(time_vector_x[time_half-1:], epoch1 , 'black')
        axs[0].vlines(time_vector_x[time_half-1], 0, 1 , 'black')
        sns.lineplot(x= all_traces.timepoint, y = all_traces.datapoint, ax=axs[1], ci=95, legend='brief')   #ci = confidence intervalls, changed to sns for this
        #axs[1].plot(time_vector_x, all_traces[index])  #if you want to plot the same with plt but without confidence intervalls
    fig.suptitle(title, fontsize=22)
    plt.legend(labels=fly_ID_list)
    
    if save == True:
        saved_plot_name = os.path.join(saved_plots_dir, '%s_Averaged_ROI_multiple_flies_%s' % (date, x_mode))    
        fig.savefig(saved_plot_name)
        
#%%plot raw traces

def plot_raw_traces(pickle_list, threshold, x_mode, saved_plots_dir, save=False):

    def whole_stim_trace (single_stim, repeats):
        if repeats < 1:
            raise Exception('invalid repeats', repeats)
        
        whole_stim = np.array([0]*59)
        for i in range (repeats):
            whole_stim = np.concatenate((whole_stim, single_stim))
        return whole_stim

    for index, file in enumerate(pickle_list):
        data, title = load_data_function(file, threshold, x_mode)
        if len(data) == 0: #if there are no reliable ROIs in the recording
            continue
        fig, axs = plot_params (title, 'frames', 'f')

        for roi in data:
            raw_traces = roi.raw_trace
            single_stim = roi.stim_trace
            #frame_rate = roi2.imaging_info['frame_rate']
            #frame_number = roi2.imaging_info['frame_timings'].size
            #total_time = 1/frame_rate * frame_number
            #x =  np.linspace(0, total_time, frame_number)

            whole_stim = whole_stim_trace(single_stim, 5)
        
            axs[1].plot(raw_traces)
        axs[0].plot(whole_stim, 'black')
        if save == True:
            saved_plot_name = os.path.join(saved_plots_dir, '%s_raw_ROIs_%s'%(roi.experiment_info['MovieID'], x_mode))
            fig.savefig(saved_plot_name)
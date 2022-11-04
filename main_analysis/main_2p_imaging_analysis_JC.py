# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:27:10 2021

@author: smolina and Burak Gur


GENERAL DISCLAIMER : the code is optimized for "2pstim" stimulation currently running in ULTIMA
For make it compatible for the "C++ stimulation" in the INVESTIGATOR and for "Pystim" in ULTIMA,
two functions in "stim_functions" still need to be implemented: readStimInformation and readStimOut

"""
#%%
import os
from time import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import warnings
from core_functions_2p_imaging_analysis_JC import load_movie, get_stim_xml_params,organize_extraction_params,get_epochs_identity, calculate_SNR_Corr,plot_roi_masks,conc_traces,interpolate_signal
from roi_selection_functions import run_ROI_selection
from roi_class_JC import generate_ROI_instances, separate_trials_ROI, get_masks_image
from core_functions_general import saveWorkspace
from darkest_pixel_background import extract_im_background
import ROI_mod
import process_mov_core_JC as pmc
import stripe_functions_JC as sf

#%% Messages to developer

print('Message to developer: ')
print('Pack the user parameters un a run-script')

#%% User parameters

# Fly-specific selection parameters
Age = '5'
Sex = 'f'
Genotype = 'Tm9GC6f_Wnt10tdTOM'
experiment = 'Tm9GC6f_Wnt10tdTOM'                  #'Mi1_GluCla_Mi1_suff_exp'
save_folder_geno = 'ExpLine' #'ExpLine'
current_date = date(2022, 2, 22)
fly_n = 2
t_series_number = 2
current_exp_ID = f'{current_date.strftime("%Y_%m_%d")}_fly{fly_n}' #e.g '2022_03_18_fly1'
current_t_series = f'TSeries-{current_date.strftime("%m%d%Y")}_fly{fly_n}-{str(t_series_number).zfill(3)}'
                    # e.g 'TSeries-03182022_fly2-001' or name of TSeries



# ROI selection/extraction parameters
time_series_stack = f'{current_t_series}_Ch2_reg.tif' # A tif stack.
roi_extraction_type = 'manual' #  'transfer' 'manual' 'cluster_analysis'
transfer_type = 'minimal' # 'minimal' (so far the single option)
transfer_data_name = f'{current_exp_ID}_{roi_extraction_type}.pickle'

use_avg_data_for_roi_extract = False
use_other_series_roiExtraction = False # ROI selection video
roiExtraction_tseries = current_t_series


deltaF_method = 'mean' # 'mean'
df_first = True # If df_f should be done BEFORE trial averaging, JC: False is not working correctly-2022_02_02!!!
int_rate = 10
bg_subtraction_method = 'selected_bg' # 'darkest_point'

# choose which type of stimulus to analyse (JC)
stimulus_type = 'standing_stripes'
# '5_sec_FFF'
# 'White_Noise'
# 'standing_stripes'

# Saving options
save_data = True

#%% Auto-setting of some other directories
dataFolder = r'U:\Dokumente\PhD\2p_data\Ultima'
#dataFolder = r'X:\2p_raw_data\Ultima\to_analyze'
initialDirectory = os.path.join(dataFolder, experiment)
alignedDataDir = os.path.join(initialDirectory,
                              'rawData\\alignedData')
stimInputDir = os.path.join(initialDirectory, 'stimulus_types')
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data')
summary_save_dir = os.path.join(alignedDataDir,
                                '_summaries')
trash_folder = os.path.join(dataFolder, 'Trash')

#%% Auto-setting of some variables
dataDir = os.path.join(alignedDataDir, current_exp_ID, current_t_series)
current_movie_ID = current_exp_ID + '-' + current_t_series
experiment_conditions = {'Genotype' : Genotype, 'Age': Age, 'Sex' : Sex,
                         'FlyID' : current_exp_ID, 'MovieID': current_movie_ID}

#%% Load of aligned data
dataDir = os.path.join(alignedDataDir, current_exp_ID, current_t_series)
time_series = load_movie(dataDir,time_series_stack)
mean_image = time_series.mean(0)
#imgplot = plt.imshow(mean_image, cmap='hot')
                                                                                #imgplot = plt.imshow(mean_image,cmap="hot")

#%% Metadata extraction (from xml, stimulus input and stimulus output file) 
imaging_information,stimulus_information, stimType, rawStimData,stimInputFile = get_stim_xml_params(dataDir, stimInputDir)

#%% Epochs sorting and identity assignment. Adding info to "stimulus_information"
if stimulus_type == 'standing_stripes':
    stimulus_information = sf.get_epochs_for_stripes(imaging_information, stimulus_information, stimType, rawStimData)

else:
    stimulus_information = get_epochs_identity(imaging_information,stimulus_information,stimType, rawStimData,stimInputFile)
    #baseline epoch 0 but what if it is not baseline? -JC

#%%  Getting some more parameters

extraction_params = organize_extraction_params(roi_extraction_type,current_t_series=current_t_series,current_exp_ID=current_exp_ID,
                               alignedDataDir=alignedDataDir,stimInputDir=stimInputDir,use_other_series_roiExtraction = use_other_series_roiExtraction,
                               use_avg_data_for_roi_extract = use_avg_data_for_roi_extract,roiExtraction_tseries=roiExtraction_tseries,
                               transfer_data_n = transfer_data_name,transfer_data_store_dir = saveOutputDir,transfer_type = transfer_type,
                               imaging_information=imaging_information,experiment_conditions=experiment_conditions)
                               #for manual only get the type? yes -JC

analysis_params = {'deltaF_method': deltaF_method, 'df_first': df_first}

#%% ROI selection    
ROI_selection_dict = run_ROI_selection(extraction_params,time_series_stack,stimulus_information, imaging_information,image_to_select=mean_image)

#%%  Creation of ROI objects of a class
if ROI_selection_dict['rois'] == None:
    del ROI_selection_dict['rois']
    rois = generate_ROI_instances(ROI_selection_dict,
                                          mean_image, 
                                          experiment_info = experiment_conditions, 
                                          imaging_info =imaging_information)


#%% Store relevant information in each roi
for roi in rois:
    roi.extraction_params = extraction_params
    if roi_extraction_type == 'transfer': # Update transferred ROIs
        roi.experiment_info = experiment_conditions
        roi.imaging_info = imaging_information
        for param in analysis_params.keys():
            roi.analysis_params[param] = analysis_params[param]
    else:
        roi.analysis_params= analysis_params

#%% Plotting ROIs and properties
if save_data:
    figure_save_dir = os.path.join(dataDir, 'Results')
else: 
    figure_save_dir = trash_folder

if not os.path.exists(figure_save_dir):
    os.mkdir(figure_save_dir)

roi_image = get_masks_image(rois)
plot_roi_masks(roi_image,mean_image,len(rois),
                current_movie_ID,save_fig=True,
                save_dir=figure_save_dir,alpha=0.4)  
#%%  Background substraction

if bg_subtraction_method == 'darkest_point':
    foreground, background = extract_im_background(time_series, time_series) #JC: changed background substraction method to darkest pixels -2022_02_01
    background_bool = background.astype(np.bool)    #JC: convert the 1,0 array to Bool (should have worked before but didn't for me, so I changed it)
    time_series = np.transpose(np.subtract(np.transpose(time_series),

                                        time_series[:,background_bool].mean(axis=1))) #JC: useing darkest pixel
elif bg_subtraction_method == 'selected_bg':    
    time_series = np.transpose(np.subtract(np.transpose(time_series),
                                        time_series[:,ROI_selection_dict['bg_mask']].mean(axis=1))) #JC: using selected bg
else:
    warnings.warn('No background subtraction done!!!')

#use bg_mask to substract values from there from the whole file -JC
#%%  
if stimulus_type == '5_sec_FFF':
    # Data sorting (epochs sorting) + Trial averaging (TA) + deltaF/f
    # ROI trial separated responses
    (wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs,respTraces_allTrials_ROIs_raw,
    baselineTraces_allTrials_ROIs) = \
        separate_trials_ROI(time_series,rois,stimulus_information,
                                imaging_information['frame_rate'],moving_avg = True, bins = 3,
                                df_method = analysis_params['deltaF_method'],df_first = df_first)
    #%%  SNR and reliability
    #SNR baseline needs tp be from raw data not df/f data! -JC
    baseTraces_SNR = baselineTraces_allTrials_ROIs
    if stimulus_information['random'] == 2:
        epoch_to_exclude = None
        baseTraces_SNR = respTraces_allTrials_ROIs_raw.copy()
    elif stimulus_information['random'] == 0:
        epoch_to_exclude = stimulus_information['baseline_epoch']
    else:
        epoch_to_exclude = None

    [SNR_rois, corr_rois] = calculate_SNR_Corr(baseTraces_SNR,
                                                respTraces_allTrials_ROIs_raw,rois,
                                                epoch_to_exclude=None)


    list(map(lambda roi: roi.appendStimInfo(stimulus_information), rois))
    list(map(lambda roi: roi.findMaxResponse_all_epochs(), rois))
    list(map(lambda roi: roi.setSourceImage(mean_image), rois))

    #Data interpolation (to 10 hz)
    print('Seb, split concatenation from interpolation.It does not make sense for some stimuli to concatenate')
    for roi in rois:
        roi.int_whole_trace_all_epochs = roi.whole_trace_all_epochs.copy()
        roi.int_stim_trace = roi.whole_trace_all_epochs.copy()  #difference? -JC
        #JC: whole_trace_all_epochs has mean of trials (df/f) for each frame
        #    for the respective ROI for each epoch stored.
        #    If movingaverage = True, traces were convolved with a certain bin size

        for idx, epoch in enumerate(list(range(0,roi.stim_info['EPOCHS']))): #Seb: epochs_number --> EPOCHS, JC: changed beginning to 0 so first eppch is included
                stimulus_dur = roi.stim_info['duration'][idx]   #JC: needed for not had coded interpolation
                roi.int_whole_trace_all_epochs[epoch] = interpolate_signal(roi.int_whole_trace_all_epochs[epoch], 
                                                    roi.imaging_info['frame_rate'], 
                                                    int_rate, 'data', stimulus_dur)
                roi.int_stim_trace[epoch] = interpolate_signal(roi.int_stim_trace[epoch], 
                                                    roi.imaging_info['frame_rate'], 
                                                    int_rate, 'stim', stimulus_dur)
                roi.int_rate = int_rate

    #ROI concatenation
    rois = conc_traces(rois, interpolation = True, int_rate = int_rate)

elif stimulus_type == 'standing_stripes':
    #df/f calculations with f0 from the inter stimulus intervall (last half before stim)
    sf.get_df_inter_stim_intervall (time_series,rois, stimulus_information)
    
    #trial averaging
    (wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs,
    respTraces_allTrials_ROIs_raw, baselineTraces_allTrials_ROIs) = \
                sf.trial_averaging_stripes (rois, stimulus_information)
    

    #SNR and reliability
    (_, _, respTraces_raw_SNR, baselineTraces_SNR_raw) = \
                sf.trial_averaging_stripes (rois, stimulus_information, df_true =False)
    baseTraces_SNR = baselineTraces_SNR_raw
    [SNR_rois, corr_rois] = calculate_SNR_Corr(baseTraces_SNR,
                                                respTraces_raw_SNR,rois,
                                                epoch_to_exclude=18)
    #plot SNR and reliability
    sf.plot_SNR_density (rois, figure_save_dir, save_option=save_data)
    
    # store info in rois
    list(map(lambda roi: roi.appendStimInfo(stimulus_information), rois))
    list(map(lambda roi: roi.findMaxResponse_all_epochs('epoch_count'), rois))
    list(map(lambda roi: roi.setSourceImage(mean_image), rois))

    #data interpolation (to 10 Hz)
    for roi in rois:
        roi.int_whole_trace_all_epochs = roi.whole_trace_all_epochs.copy()
        roi.int_resp_trace_all_epochs = roi.resp_trace_all_epochs.copy()
        for idx, epoch in enumerate(list(range(1,(roi.stim_info['epoch_count']-1)))): #Seb: epochs_number --> 'epoch_count' (JC)
                stimulus_dur_whole = roi.stim_info['bar.duration'][0] + \
                    (roi.stim_info['bar.duration'][0])
                stimulus_dur_resp = roi.stim_info['bar.duration'][0]
                roi.int_whole_trace_all_epochs[epoch] = interpolate_signal(roi.int_whole_trace_all_epochs[epoch], 
                                                    roi.imaging_info['frame_rate'], 
                                                    int_rate, 'data', stimulus_dur_whole)
                roi.int_resp_trace_all_epochs[epoch] = interpolate_signal(roi.int_whole_trace_all_epochs[epoch], 
                                                    roi.imaging_info['frame_rate'], 
                                                    int_rate, 'data', stimulus_dur_resp)

                roi.int_rate = int_rate
        #plotting trial average for all epochs
        int_whole_traces = roi.int_whole_trace_all_epochs
        roi_id = roi.uniq_id
        sf.plot_averaged_trials (int_whole_traces, roi_id, figure_save_dir)
    
    #do thresholding only later
    ##used_rois = sf.thresholding (rois,threshold=0.7, SNR_threshold=0.8)
    fly_date = rois[0].experiment_info['MovieID'] 
    sf.plot_sorted_traces(rois, figure_save_dir, save_data, fly_date)
    sf.plot_centered_traces_and_heatmap (rois, figure_save_dir, fly_date, save_data)

elif stimulus_type == 'White_Noise':
    #Seb: generating the ternary noise stimulus
    choiseArr = [0,0.5,1]
    x = 16
    y = 16  #JC: changed from 1 to 16 because I have 16X16X10000 stimulus
    z= 10000 # z- dimension (here frames presented over time)
    np.random.seed(54378) #Fix seed. Do not ever change before calling this from stim_output_file
    stim= np.random.choice(choiseArr, size=(z,x,y))

    # ROI raw signals
    for iROI, roi in enumerate(rois):
        roi.raw_trace = time_series[:,roi.mask].mean(axis=1)
        roi.wn_stim = stim

    # Append relevant information and calculate some parameters
    list(map(lambda roi: roi.appendStimInfo(stimulus_information), rois))
    list(map(lambda roi: roi.setSourceImage(mean_image), rois))


    #%% White noise analysis

    rois = ROI_mod.reverse_correlation_analysis(rois, noise_type='grit') #JC: added grit option
    #final_rois = rois

#%%  Single Fly summary plot (includes TA trace per epoch, basic response calculations)


#%%  Save data pickle file 
if save_data:
    os.chdir(dataFolder) # Seb: data_save_vars.txt file needs to be there    
    varDict = locals()
    pckl_save_name = ('%s_%s' % (current_movie_ID, extraction_params['type']))
    saveOutputDir = os.path.join(saveOutputDir, varDict['varDict']['stimulus_information']['stim_name'][:-4]) #Seb: experiment_folder/analyzed_data/stim_name/genotype_folder
    if not os.path.exists(saveOutputDir):
            os.mkdir(saveOutputDir) # Seb: creating stim_folder
    saveOutputDir = os.path.join(saveOutputDir,save_folder_geno)
    if not os.path.exists(saveOutputDir):
            os.mkdir(saveOutputDir) # Seb: creating genotype_folder
    saveWorkspace(saveOutputDir,pckl_save_name, varDict, 
               varFile='data_save_vars.txt',extension='.pickle')

    print('\n\n%s saved...\n\n' % pckl_save_name)
else:
    print('Pickle data not created')



#%% Temporary
final_rois = rois
final_roi_image = get_masks_image(final_rois)

#%%

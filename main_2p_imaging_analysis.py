# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:27:10 2021

@author: smolina and Burak Gur


GENERAL DISCLAIMER : the code is optimized for "2pstim" stimulation currently running in ULTIMA
For make it compatible for the "C++ stimulation" in the INVESTIGATOR and for "Pystim" in ULTIMA,
two functions in "stim_functions" still need to be implemented: readStimInformation and readStimOut

"""

import os
import matplotlib.pyplot as plt
import numpy as np
from core_functions_2p_imaging_analysis import load_movie, get_stim_xml_params,organize_extraction_params,get_epochs_identity, calculate_SNR_Corr,plot_roi_masks,conc_traces,interpolate_signal
from roi_selection_functions import run_ROI_selection
from roi_class import generate_ROI_instances, separate_trials_ROI, get_masks_image
from core_functions_general import saveWorkspace

#%% Messages to developer

print('Message to developer: ')
print('Pack the user parameters un a run-script')

#%% User parameters

# Fly-specific selection parameters
experiment = 'Mi1_GluCla_Mi1_suff_exp'
current_exp_ID = '20210617_seb_fly1'
current_t_series ='TSeries-fly1-002'
Genotype = 'Mi1_GCaMP6f_Mi1_GluCla_ExpLine_GluCla_suff_exp'
save_folder_geno = 'ExpLine'
Age = '4'
Sex = 'm'


# ROI selection/extraction parameters
time_series_stack = 'Mot_corr_stack.tif'# 'Raw_stack.tif' 'Mot_corr_stack.tif' # A tif stack.
roi_extraction_type = 'manual' #  'transfer' 'manual' 'cluster_analysis'
transfer_type = 'minimal' # 'minimal' (so far the single option)
transfer_data_name = '20201213_seb_fly4-TSeries-fly4-001_manual.pickle'

use_avg_data_for_roi_extract = False
use_other_series_roiExtraction = False # ROI selection video
roiExtraction_tseries = 'TSeries-fly1-001'


deltaF_method = 'mean' # 'mean'
df_first = False # If df_f should be done BEFORE trial averaging
int_rate = 10 #10 hz extrapolation

# Saving options
save_data = False

#%% Auto-setting of some other directories
dataFolder = r'F:\SebastianFilesExternalDrive\Science\PhDAGSilies\2pData Python_data'
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
                                                                                #imgplot = plt.imshow(mean_image)
                                                                                #imgplot = plt.imshow(mean_image,cmap="hot")

#%% Metadata extraction (from xml, stimulus input and stimulus output file) 
imaging_information,stimulus_information, stimType, rawStimData,stimInputFile = get_stim_xml_params(dataDir, stimInputDir)

#%% Epochs sorting and identity assignment. Adding info to "stimulus_information"
stimulus_information = get_epochs_identity(imaging_information,stimulus_information,stimType, rawStimData,stimInputFile)


#%%  ROI selection

extraction_params = organize_extraction_params(roi_extraction_type,current_t_series=current_t_series,current_exp_ID=current_exp_ID,
                               alignedDataDir=alignedDataDir,stimInputDir=stimInputDir,use_other_series_roiExtraction = use_other_series_roiExtraction,
                               use_avg_data_for_roi_extract = use_avg_data_for_roi_extract,roiExtraction_tseries=roiExtraction_tseries,
                               transfer_data_n = transfer_data_name,transfer_data_store_dir = saveOutputDir,transfer_type = transfer_type,
                               imaging_information=imaging_information,experiment_conditions=experiment_conditions)
    
ROI_selection_dict = run_ROI_selection(extraction_params,time_series_stack,stimulus_information, imaging_information,image_to_select=mean_image)

#%%  Creation of ROI objects of a class
if ROI_selection_dict['rois'] == None:
    del ROI_selection_dict['rois']
    rois = generate_ROI_instances(ROI_selection_dict,
                                          mean_image, 
                                          experiment_info = experiment_conditions, 
                                          imaging_info =imaging_information)

    
#%%  Background substraction
time_series = np.transpose(np.subtract(np.transpose(time_series),
                                       time_series[:,ROI_selection_dict['bg_mask']].mean(axis=1)))
#%%  Data sorting (epochs sorting) + Trial averaging (TA) + deltaF/f
# ROI trial separated responses
analysis_params = {'deltaF_method': deltaF_method, 'df_first': df_first} 

(wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs,respTraces_allTrials_ROIs_raw,
 baselineTraces_allTrials_ROIs) = \
    separate_trials_ROI(time_series,rois,stimulus_information,
                               imaging_information['frame_rate'],moving_avg = True, bins = 3,
                               df_method = analysis_params['deltaF_method'],df_first = df_first)
#%%  SNR and reliability
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

list(map(lambda roi: roi.appendStimInfo(stimulus_information), rois))
list(map(lambda roi: roi.findMaxResponse_all_epochs(), rois))
list(map(lambda roi: roi.setSourceImage(mean_image), rois))

#%%  Data interpolation (to 10 hz)
print('Seb, whole_trace_all_epoch last too long? Expected 10, received 100')
for roi in rois:
    roi.int_whole_trace_all_epochs = roi.whole_trace_all_epochs.copy()
    roi.int_stim_trace = roi.whole_trace_all_epochs.copy()

    for idx, epoch in enumerate(list(range(1,roi.stim_info['EPOCHS']))): #Seb: epochs_number --> EPOCHS
            roi.int_whole_trace_all_epochs[epoch] = interpolate_signal(roi.int_whole_trace_all_epochs[epoch], 
                                                   roi.imaging_info['frame_rate'], 
                                                   int_rate)
            curr_stim = np.zeros((1,len(roi.whole_trace_all_epochs[epoch])))[0]
            curr_stim = curr_stim + idx
            roi.int_stim_trace[epoch] = interpolate_signal(curr_stim, 
                                                   roi.imaging_info['frame_rate'], 
                                                   int_rate,int_time = None)
            roi.int_rate = int_rate

#%%  ROI concatenation
rois = conc_traces(rois, interpolation = False, int_rate = int_rate)
#%%  Raw data plot (whole stimulus and response traces for all ROIs, colormaps of masks with SNR-reliability-reponse quality)
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

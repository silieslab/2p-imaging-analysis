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
from core_functions_2p_imaging_analysis import load_movie, get_stim_xml_params,organize_extraction_params,get_epochs_identity
from roi_selection_functions import run_ROI_selection
from roi_class import generate_ROI_instances

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
    
(cat_masks, cat_names, roi_masks, all_rois_image, rois, threshold_dict) = run_ROI_selection(extraction_params,time_series_stack,stimulus_information, imaging_information,image_to_select=mean_image)

#%%  Creation of ROI class
if rois == None:
    del rois
    rois = generate_ROI_instances(roi_masks, cat_masks, cat_names,
                                          mean_image, 
                                          experiment_info = experiment_conditions, 
                                          imaging_info =imaging_information)

#%% Store inforamtion in each roi
for roi in rois:
    roi.extraction_params = extraction_params
    if roi_extraction_type == 'transfer': # Update transferred ROIs
        roi.experiment_info = experiment_conditions
        roi.imaging_info = imaging_information
    
#%%  Background substraction
#%%  Data sorting (epochs sorting, including subepochs trigger by tau)
#%%  Trial averaging (TA)
#%%  deltaF/f
#%%  SNR and reliability
#%%  Quality index
#%%  Data interpolation (to 10 hz)
#%%  Basic response calculation per epoch (max, min, integral, time to peak, decay rate, correlation)
#%%  Raw data plot (whole stimulus and response traces for all ROIs, colormaps of masks with SNR-reliability-reponse quality)
#%%  Single Fly summary plot (includes TA trace per epoch, basic response calculations)
#%%  Save data pickle file / summary plots

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:52:56 2020

@author: burakgur
modified by sebasto_7
modified by JacquelineC
"""
# %% Importing packages
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys

code_path = r'\\fs02\jcornean$\Dokumente\python_seb_lumgrating_course\msc_course_2p_analysis\for_msc_course'
#code_path = r'C:\Users\ASUS\Dokumente\python_seb_lumgrating_course\2p_analysis\for_msc_course' # It must be change for the path where the code is in each PC
sys.path.insert(0, code_path) 
os.chdir(code_path)
import ROI_mod_reduced_msc_course as ROI_mod
from core_functions_reduced_msc_course import saveWorkspace
import process_mov_core_reduced_msc_course as pmc

#%% Setting the directories

#dataFolder = r'G:\SebastianFilesExternalDrive\Science\PhDAGSilies\2pData Python_data'
#dataFolder = r'Z:\2p data Ultima\2pData Python_data'
#dataFolder = r'C:\Users\ASUS\Dokumente\python_seb_lumgrating_course\2p_analysis\Data_Sebastian'
dataFolder = r'\\fs02\jcornean$\Dokumente\python_seb_lumgrating_course\2p_analysis\Data_Sebastian'
save_data = True # choose True or False
save_raw_ROIs = True
# Plot related
plot_roi_summ = True # choose True or False
#%% Parameters to adjust
plt.close('all')

# Experimental parameters
experiment = 'Mi1_GluCla_Mi1_suff_exp'
current_exp_ID = '20210217_seb_fly4'
current_t_series ='TSeries-fly4-003'
Genotype = 'Mi1_GCaMP6f_Mi1_GluCla_PosCnt_GluCla_suff_exp'
save_folder_geno = 'PosCnt'
Age = '3'
Sex = 'f'

time_series_stack = 'TSeries-fly4-003_Ch2_reg.tif'# 'Raw_stack.tif' 'Mot_corr_stack.tif'

# Analysis parameters
analysis_type = 'lumgratings'

# Auto-setting of some other directories
initialDirectory = os.path.join(dataFolder, experiment)

alignedDataDir = os.path.join(initialDirectory,
                              'rawData\\alignedData')
stimInputDir = os.path.join(initialDirectory, 'stimulus_types')
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data')
summary_save_dir = os.path.join(alignedDataDir,
                                '_summaries')
trash_folder = os.path.join(dataFolder, 'Trash')

# ROI selection/extraction parameters
extraction_type = 'manual' # 'SIMA-STICA' 'transfer' 'manual'  Seb: SIMA-STICA for extracting clusters. Tranfer for tranfering the ROI clusters from one sequence to other
transfer_type = 'minimal' # 'minimal' 'predefined'
transfer_data_name = '20210217_seb_fly4-TSeries-fly4-003_manual.pickle'

use_avg_data_for_roi_extract = False
use_other_series_roiExtraction = False # ROI selection video
roiExtraction_tseries = 'TSeries-fly4-003'

#%% Get the stimulus and imaging information
dataDir = os.path.join(alignedDataDir, current_exp_ID, current_t_series)

(time_series, stimulus_information,imaging_information) = \
    pmc.pre_processing_movie (dataDir,stimInputDir,time_series_stack)
mean_image = time_series.mean(0)
current_movie_ID = current_exp_ID + '-' + current_t_series
if save_data:
    figure_save_dir = os.path.join(dataDir, 'Results')
else: 
    figure_save_dir = trash_folder

if not os.path.exists(figure_save_dir):
    os.mkdir(figure_save_dir)
experiment_conditions = \
    {'Genotype' : Genotype, 'Age': Age, 'Sex' : Sex,
     'FlyID' : current_exp_ID, 'MovieID': current_movie_ID}

#%% Define analysis/extraction parameters and run region selection
#   generate ROI objects.

# Organizing extraction parameters
if transfer_type == 'predefined':
    transfer_type = analysis_type
    
extraction_params = \
    pmc.organize_extraction_params(extraction_type,
                               current_t_series=current_t_series,
                               current_exp_ID=current_exp_ID,
                               alignedDataDir=alignedDataDir,
                               stimInputDir=stimInputDir,
                               use_other_series_roiExtraction = use_other_series_roiExtraction,
                               use_avg_data_for_roi_extract = use_avg_data_for_roi_extract,
                               roiExtraction_tseries=roiExtraction_tseries,
                               transfer_data_n = transfer_data_name,
                               transfer_data_store_dir = saveOutputDir,
                               transfer_type = transfer_type,
                               imaging_information=imaging_information,
                               experiment_conditions=experiment_conditions)
        
    
analysis_params = {'deltaF_method': 'mean',
                   'analysis_type': analysis_type} 
#%%
# Select/extract ROIs
(cat_masks, cat_names, roi_masks, all_rois_image, rois,
threshold_dict) = \
    pmc.run_ROI_selection(extraction_params,time_series_stack,image_to_select=mean_image)

#save extracted ROIs in a .pickle file
if save_raw_ROIs:
    os.chdir(dataFolder) # Seb: data_save_vars.txt file needs to be there    
    varDict = locals()
    pckl_save_name = ('%s_%s' % (current_movie_ID, extraction_params['type'], 'raw_ROI'))
    saveOutputDir = os.path.join(saveOutputDir, varDict['varDict']['stimulus_information']['stim_name'][:-4]) #Seb: experiment_folder/analyzed_data/stim_name/genotype_folder
    if not os.path.exists(saveOutputDir):
            os.mkdir(saveOutputDir) # Seb: creating stim_folder
    saveOutputDir = os.path.join(saveOutputDir,save_folder_geno)
    if not os.path.exists(saveOutputDir):
            os.mkdir(saveOutputDir) # Seb: creating genotype_folder
    saveWorkspace(saveOutputDir,pckl_save_name, varDict, 
               varFile='roi_save_vars.txt',extension='.pickle')

    print('\n\n%s saved...\n\n' % pckl_save_name)
else:
    print('Pickle data not created')

# Get mask for background, needed for bg subtration later
for idx, cat_name in enumerate(cat_names):
    if cat_name.lower() == 'bg\r': # Seb: added '\r' for debugging mode
        bg_mask = cat_masks[idx]
        continue
    elif cat_name.lower() == 'bg': 
        bg_mask = cat_masks[idx]
        continue

# Generate ROI_bg instances
if rois == None:
    del rois
    rois = ROI_mod.generate_ROI_instances(roi_masks, cat_masks, cat_names,
                                          mean_image, 
                                          experiment_info = experiment_conditions, 
                                          imaging_info =imaging_information)

# We can store the parameters inside the objects for further use
for roi in rois:
    roi.extraction_params = extraction_params
    if extraction_type == 'transfer': # Update transferred ROIs
        roi.experiment_info = experiment_conditions
        roi.imaging_info = imaging_information
        for param in analysis_params.keys():
            roi.analysis_params[param] = analysis_params[param]
    else:
        roi.analysis_params= analysis_params

#%% # BG subtraction
time_series = np.transpose(np.subtract(np.transpose(time_series),
                                       time_series[:,bg_mask].mean(axis=1)))
print('\n Background subtraction done...')

#%% # ROI trial separated responses
(wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs,
 baselineTraces_allTrials_ROIs) = \
    pmc.separate_trials_ROI_v4(time_series,rois,stimulus_information,
                               imaging_information['frame_rate'],
                               df_method = analysis_params['deltaF_method'])

#%% Append relevant information and calculate some parameters
list(map(lambda roi: roi.appendStimInfo(stimulus_information), rois))
list(map(lambda roi: roi.findMaxResponse_all_epochs(), rois))
list(map(lambda roi: roi.setSourceImage(mean_image), rois))

#%% df/f calculation and trial average 
(_, respTraces_SNR, baseTraces_SNR) = \
    pmc.separate_trials_ROI_v4(time_series,rois,stimulus_information,
                               imaging_information['frame_rate'],
                               df_method = analysis_params['deltaF_method'],
                               df_use=False)
#%% SNR and reliability
if stimulus_information['random'] == 2:
    epoch_to_exclude = None
    baseTraces_SNR = respTraces_SNR.copy()
    # baseTraces_SNR[]
elif stimulus_information['random'] == 0:
    epoch_to_exclude = stimulus_information['baseline_epoch']
else:
    epoch_to_exclude = None

[SNR_rois, corr_rois] = pmc.calculate_SNR_Corr(baseTraces_SNR,
                                               respTraces_SNR,rois,
                                               epoch_to_exclude=None)

#%% Thresholding
if threshold_dict is None:
    print('No threshold used, all ROIs will be retained')
    thresholded_rois = rois
else:
    print('Thresholding ROIs')
    thresholded_rois = ROI_mod.threshold_ROIs(rois, threshold_dict)

final_rois = thresholded_rois
final_roi_image = ROI_mod.get_masks_image(final_rois)

#%% Plotting ROIs and properties
pmc.plot_roi_masks(final_roi_image,mean_image,len(final_rois),
                   current_movie_ID,save_fig=True,
                   save_dir=figure_save_dir,alpha=0.4)

#%% Run desired analyses for different types
final_rois = pmc.run_analysis(analysis_params,final_rois,experiment_conditions,
                              imaging_information,summary_save_dir,
                              save_fig=True,fig_save_dir = figure_save_dir,
                              exp_ID=('%s_%s' % (current_movie_ID,
                                                 extraction_params['type'])))

#%% Make figures for experiment summary
images = []
(properties, colormaps, vminmax, data_to_extract) = \
    pmc.select_properties_plot(final_rois , analysis_params['analysis_type'])
for prop in properties:
    images.append(ROI_mod.generate_colorMasks_properties(final_rois, prop))
pmc.plot_roi_properties(images, properties, colormaps, mean_image,
                        vminmax,current_movie_ID, imaging_information['depth'],
                        save_fig=True, save_dir=figure_save_dir,figsize=(8, 6),
                        alpha=0.5)
final_roi_data = ROI_mod.data_to_list(final_rois, data_to_extract)
rois_df = pd.DataFrame.from_dict(final_roi_data)

pmc.plot_df_dataset(rois_df,data_to_extract,
                    exp_ID=('%s_%s' % (current_movie_ID,
                                       extraction_params['type'])),
                    save_fig=True, save_dir=figure_save_dir)
plt.close('all')

#%% PART 4: Save data
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
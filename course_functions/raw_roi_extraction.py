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
#code_path = r'\\fs02\smolina$\Dokumente\msc_course_2p_analysis\for_msc_course'
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
#dataFolder = r'Z:\2p data Ultima\2pData Python_data'
save_data = True # choose True or False
save_raw_ROIs = True
# Plot related
plot_roi_summ = True # choose True or False
#%% Parameters to adjust
plt.close('all')

# Experimental parameters
experiment = 'Mi1_GluCla_Mi1_suff_exp'
current_exp_ID = '20210217_seb_fly4'
current_t_series ='TSeries-fly4-002'
Genotype = 'Mi1_GCaMP6f_Mi1_GluCla_PosCnt_GluCla_suff_exp'
save_folder_geno = 'PosCnt'
Age = '3'
Sex = 'f'

time_series_stack = 'TSeries-fly4-002_Ch2_reg.tif'# 'Raw_stack.tif' 'Mot_corr_stack.tif'

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
transfer_data_name = '20210217_seb_fly4-TSeries-fly4-002_manual.pickle'

use_avg_data_for_roi_extract = False
use_other_series_roiExtraction = False # ROI selection video
roiExtraction_tseries = 'TSeries-fly4-002'

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
    pckl_save_name = f'{current_movie_ID}_{extraction_params["type"]}_extracted_rois'
    saveOutputDir = os.path.join(alignedDataDir, 'extracted_rois')
    if not os.path.exists(saveOutputDir):
            os.mkdir(saveOutputDir) # Seb: creating stim_folder 

    saveWorkspace(saveOutputDir,pckl_save_name, varDict, 
               varFile='roi_save_vars.txt',extension='.pickle')

    print('\n\n%s saved...\n\n' % pckl_save_name)
else:
    print('Pickle data not created')
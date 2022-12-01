 #-*- coding: utf-8 -*-

"""
Created on Thu July 15 08:29:00 2022

@author: Sebastian Molina-Obando

Run script for the main analysis of calcium imaging data
"""
#Importing analysis function
from main_analysis.main_2p_imaging_analysis import main_2p_imaging_analysis
#%% Please complete the following information
userParams = {}
# Fly-specific selection parameters
#TODO Add an option to load this information from a metadata file

userParams['experiment'] = 'LC11-BL68362-2x-GCaMP6f'
userParams['current_exp_ID'] = '20221111-fly2'
userParams['current_t_series'] ='TSeries-fly2-012'
userParams['genotype'] = 'LC11-splitGal4-2x-GCaMP6f'
userParams['save_folder_geno'] = 'ExpLine'
userParams['stim_name'] = 'DS100-100WB'
userParams['age'] = '3'
userParams['sex'] = 'f'

# Choose ROI selection/extraction parameters
userParams['time_series_stack'] = f"{userParams['current_t_series']}_Ch2_reg.tif" # Name of your motion-aligned tif stack.
userParams['roi_extraction_type'] = 'manual' #  'transfer' 'manual' 'cluster_analysis'
userParams['transfer_type'] = 'minimal' # 'minimal' (so far the single option)
userParams['transfer_TSeries'] = 'TSeries-fly4-001' # Choose the other TSeries you want to tranfer the ROIs from
userParams['transfer_data_name'] = f"{userParams['current_exp_ID']}-{userParams['transfer_TSeries']}_{userParams['roi_extraction_type']}.pickle"

userParams['use_avg_data_for_roi_extract'] = False
userParams['use_other_series_roiExtraction'] = False # ROI selection video
userParams['roiExtraction_tseries'] = userParams['current_t_series']


userParams['deltaF_method'] = 'mean' # 'mean'
userParams['df_first'] = True # If df_f should be done BEFORE trial averaging. False option has bugs
userParams['int_rate'] = 10 # Rate to interpolate the data

# Choose which type of stimulus to analyse (JC)
userParams['stimulus_type'] = 'general'
# 'general' # For general pre analysis of any stimulus. (stimulus agnostic)
# 'White_Noise' # Specifically for white noise stimulation (this is a temporary option that will be moved to stim-specific postanylsis function)

# Saving options
save_data = True #For saving analyzed data in pickle files AND general plots (non stimulus specific plots)
dataFolder = r'D:\2pData Ultima'  # Main folder where the folder for the userParams['experiment'] is located

#%% Calling and running the actual analysis function
main_2p_imaging_analysis(userParams, dataFolder, save_data)
 #-*- coding: utf-8 -*-

"""
Created on Thu July 15 08:29:00 2022

@author: smolina

Run script for the main analysis of calcium imaging data
"""
# Adding code to the path
import sys
code_path = r'\\fs02\smolina$\Dokumente\GitHub\2p-imaging-analysis\main_analysis' # It must be change for the path where the code is in each PC
sys.path.insert(0, code_path) 
code_path = r'\\fs02\smolina$\Dokumente\GitHub\2p-imaging-analysis\post_analysis' # It must be change for the path where the code is in each PC
sys.path.insert(0, code_path) 

#Importing analysis function
from main_2p_imaging_analysis import main_2p_imaging_analysis

#%% Please complete the followin information
userParams = {}
# Fly-specific selection parameters
userParams['experiment'] = 'Mi1_GluCla_Mi1_suff_exp'
userParams['current_exp_ID'] = '20201213_seb_fly1'
userParams['current_t_series'] ='TSeries-fly1-011'
userParams['Genotype'] = 'Mi1_GCaMP6f_Mi1_GluCla_ExpLine_GluCla_suff_exp'
userParams['save_folder_geno'] = 'ExpLine'
userParams['Age'] = '6'
userParams['Sex'] = 'f'

# Choose ROI selection/extraction parameters
userParams['time_series_stack'] = 'Mot_corr_stack.tif' # A tif stack. (E.g. # 'Raw_stack.tif' 'Mot_corr_stack.tif' '{current_t_series}_Ch2_reg.tif')
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
userParams['stimulus_type'] = 'White_Noise'
# '5_sec_FFF'
# 'White_Noise'

# Saving options
save_data = True
dataFolder = r'Z:\2p data Ultima\2pData Python_data'  # Main folder where the folder for the userParams['experiment'] is located

#%% Calling and running the actual analysis function
main_2p_imaging_analysis(userParams, dataFolder, save_data)
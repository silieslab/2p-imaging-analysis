# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:27:10 2021

@author: smolina and Burak Gur


GENERAL DISCLAIMER : the code is optimized for "2pstim" stimulation currently running in ULTIMA
For make it compatible for the "C++ stimulation" in the INVESTIGATOR and for "Pystim" in ULTIMA,
two functions in "stim_functions" still need to be implemented: readStimInformation and readStimOut

"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from core_functions_2p_imaging_analysis import load_movie, get_stim_xml_params

#%% User parameters
experiment = 'Mi1_GluCla_Tm3_suff_exp'
current_exp_ID = '20210111_jv_fly4'
current_t_series ='TSeries-001'
Genotype = 'Mi1_GCaMP6f_Tm3_GluCla_PosCnt_GluCla_suff_exp'
save_folder_geno = 'PosCnt'
Age = '2'
Sex = 'm'

save_data = True
time_series_stack = 'Mot_corr_stack.tif'# 'Raw_stack.tif' 'Mot_corr_stack.tif'

#%% Auto-setting of some other directories
dataFolder = r'G:\SebastianFilesExternalDrive\Science\PhDAGSilies\2pData Python_data'
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

#%% Load of aligned data
dataDir = os.path.join(alignedDataDir, current_exp_ID, current_t_series)
time_series = load_movie(dataDir,time_series_stack)
mean_image = time_series.mean(0)

#%% Metadata extraction (from xml, stimulus input and stimulus output file) 
imaging_information,stimulus_information, stimType, rawStimData,stimInputFile = get_stim_xml_params(dataDir, stimInputDir)

#%% Epochs sorting and identity assignment. Adding info to "stimulus_information"
stimulus_information = get_epochs_identity(stimulus_information,stimType, rawStimData,stimInputFile,stimInputData)


#%%  ROI selection
#%%  Background substraction
#%%  Data sorting (epochs sorting, including subepochs trigger by tau)
#%%  Trial averaging (TA)
#%%  deltaF/f
#%%  SNR and reliability
#%%  Quality index
#%%  Data interpolation (to 10 hz)
#%%  Basic response calculation per epoch (max, min, integral, time to peak, decay rate, correlation)
#%%  Raw data plot (whole stimulus and response traces for all ROIs, colormaps of masks with SNR-reliability-reponse quality)
#%%  Single ROI summary plot (includes TA trace per epoch, basic response calculations)
#%%  Save data pickle file

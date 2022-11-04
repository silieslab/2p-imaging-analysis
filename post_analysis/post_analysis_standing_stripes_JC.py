
# env 2pImaging python 3.9.7
# -*- coding: utf-8 -*-

'''
Created on 22.03.2022

@author: Jacqueline Cornean (JC)

Plots for two photon data after processing (interploation,
trial averaging, df/f and background substraction.
Post-analysis for Stripe stimulus

'''

#%% 
from cProfile import label
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import glob
import pandas as pd
import numpy as np
import operator
from scipy.stats import pearsonr
os.chdir(r'\\fs02\jcornean$\Dokumente\PhD\python_github\2p-imaging-analysis-branch_develop_main')
#os.chdir(r'C:\Users\ASUS\Desktop\2p-imaging-analysis-branch_develop_main')
from roi_class_JC import ROI_bg
import ROI_mod
import stripe_functions_JC as sf

#%%
experiment = 'Tm9GC6f_Wnt10tdTOM'   #change for another genotype
stim_type = 'Stripe_1sec_1secBG_5deg_ver_random_LumDec'  #change for another stimulus type
dataFolder = r'U:\Dokumente\PhD\2p_data\Ultima'
#dataFolder = r'C:\Users\ASUS\Dokumente\two_photon'
initialDirectory = os.path.join(dataFolder, experiment)
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data')
alignedDataDir = os.path.join(initialDirectory,
                              'rawData\\alignedData')

saved_data_dir = os.path.join(saveOutputDir, stim_type, 'ExpLine') #changed ExpLine to test for debugging
saved_data_path = os.path.join(saved_data_dir,'*.pickle')
pickle_list = (glob.glob(saved_data_path))  #get a list of all the pickle file paths
saved_plots_dir = os.path.join(saveOutputDir, stim_type, 'plots') #'plots'
date = '2022_08_31'
threshold = 0.7
save_option = True
single_fly = False
#%%
fly_data = []
wanted_pickle = list(range(len(pickle_list)))
for pick in wanted_pickle:
    curr_loaded_data = pickle.load(open(pickle_list[pick], 'rb'))
    fly_data.append(curr_loaded_data['rois'])

rois = []
for i in fly_data:
    for j in i:
        rois.append(j)

if single_fly == True:
    date = fly_data['rois'][0].experiment_info['MovieID']
    rois = fly_data['rois']
#%%


#sf.plot_SNR_density (rois, saved_plots_dir, save_option=save_option)
used_rois = sf.thresholding (rois,threshold=0.7, SNR_threshold=0.8)
sf.plot_sorted_traces(used_rois, saved_plots_dir, save_option, date)
sf.plot_centered_traces_and_heatmap (used_rois, saved_plots_dir, date, save_option)
used_rois_gauss = sf.make_gaussian_fit (used_rois)
print('Now visualizing FWHM')
sf.plot_fwhm_max_resp (used_rois, saved_plots_dir, date, save_option)
sf.plot_gauss_fit_trace (used_rois,saved_plots_dir, date, save_option,centered=True)

print('To Do: Sort ROI position.')



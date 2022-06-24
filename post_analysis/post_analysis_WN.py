# env 2pImaging python 3.9.7
# -*- coding: utf-8 -*-

'''
Created on 22.03.2022

@author: Jacqueline Cornean (JC)

Code for White Noise Post Analysis data, to plot RF of single ROIs
and to get the half width maximum...
This code leans on and uses part of the code of @burakgur and @smolina.
'''
#%% Importing packages
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Adding code to the path
import sys
code_path = r'\\fs02\smolina$\Dokumente\GitHub\2p-imaging-analysis\main_analysis' # It must be change for the path where the code is in each PC
sys.path.insert(0, code_path) 
code_path = r'\\fs02\smolina$\Dokumente\GitHub\2p-imaging-analysis\post_analysis' # It must be change for the path where the code is in each PC
sys.path.insert(0, code_path) 

# Importing functions
from roi_class import ROI_bg
import ROI_mod
import process_mov_core as pmc

#plt.close('all')
#%% Setting directories where the pickle files are stored
#   and where the results are saved to

experiment = 'Mi1_GluCla_Mi1_suff_exp'   #change for another genotype
stim_type = 'StimulusData_Discrete_16_16_100000_El_50ms'  #change for another stimulus type
dataFolder = r'Z:\2p data Ultima\2pData Python_data'
initialDirectory = os.path.join(dataFolder, experiment)
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data')
saved_data_dir = os.path.join(saveOutputDir, stim_type, 'ExpLine') #changed ExpLine to test for debugging
saved_data_path = os.path.join(saved_data_dir,'*.pickle')
pickle_list = (glob.glob(saved_data_path))  #get a list of all the pickle file paths
saved_plots_dir = os.path.join(saveOutputDir, stim_type, 'raw_plots') #'plots'

#%%
# define figure_save_dir and summary_save_dir
if not os.path.exists(saved_plots_dir):
    os.mkdir(saved_plots_dir) #JC creating plot folder for in the genotype folder

summary_save_dir = os.path.join(saveOutputDir, 'summary')
if not os.path.exists(summary_save_dir):
    os.mkdir(summary_save_dir)
#%% Load the data
#pick_list = pickle_list[1]
for file in pickle_list:
    fly_data = pickle.load(open(file, 'rb'))
    rois = fly_data['rois'] # 
    
    #%% Plotting the STAs
    # Plotting ROIs and properties
    #roi_im = ROI_mod.get_masks_image(rois)
    #pmc.plot_roi_masks(roi_im,mean_image,len(rois),
    #                rois.experiment_info['MovieID'],save_fig=True,
    #                save_dir=saved_plots_dir,alpha=0.4)
    
        
    # Plotting STRFs
    fig1, fig2, fig3= ROI_mod.plot_STRFs(rois, f_w=None,number=None,cmap='coolwarm')
    fig1.suptitle(rois[0].experiment_info['Genotype'], fontsize = 6, y=0.98)
    
    f1_n = 'STRFs_%s' % (rois[0].experiment_info['FlyID'])
    
    os.chdir(saved_plots_dir)
    fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
                transparent=False,dpi=300)
    #if fig2 == True:
    fig2.suptitle('Time-Space Dim1')
    fig3.suptitle('Time-Space Dim3')
    f2_n = 'STRFs_time_space1_%s' % (rois[0].experiment_info['FlyID'])
    f3_n = 'STRFs_time_space2_%s' % (rois[0].experiment_info['FlyID'])
    fig2.savefig('%s.png'% f2_n, bbox_inches='tight',
            transparent=False,dpi=300)
    fig3.savefig('%s.png'% f3_n, bbox_inches='tight',
            transparent=False,dpi=300)
    
    os.chdir(summary_save_dir)
    f1_n = 'Summary_%s' % (rois[0].experiment_info['FlyID'])
    fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
                transparent=False,dpi=300)
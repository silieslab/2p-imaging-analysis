#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:28:33 2020

@author: burakgur
modified by Seb
"""

#%%
try:
    import cPickle # For Python 2.X
except:
    import pickle as cPickle # For Python 3.X
import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress, pearsonr
from sklearn import preprocessing
from scipy import ndimage
from matplotlib import cm
import matplotlib as mpl
import seaborn as sns
from scipy import stats
import sys

code_path = r'\\fs02\jcornean$\Dokumente\python_seb_lumgrating_course\msc_course_2p_analysis\for_msc_course'
#code_path = r'\\fs02\smolina$\Dokumente\msc_course_2p_analysis\for_msc_course'
sys.path.insert(0, code_path)
#sys.path.insert(0, code_path2)  
os.chdir(code_path)

import ROI_mod_reduced_msc_course as ROI_mod
import post_analysis_core as pac

#%%# %% Load datasets and desired variables

# Initialize variables

def get_polarity_and_cat_dict():
    polarity_dict={'Mi1' : 'ON','Tm9' : 'OFF','Mi4' : 'ON',
            'L1_' : 'OFF','L2_' : 'OFF',
            'L3_': 'OFF','Tm1': 'OFF',
            'Tm3' : 'ON'}


    cat_dict={
            'L1_' : 'M1',
            'Tm3' : 'M9',
            'L5_' : 'M1',
            'Mi1' : 'Ax',
            'Mi1_ExpLine' : 'Ax',
            'Mi1_PosCnt' : 'Ax'}
    return polarity_dict, cat_dict

#%% JC: put some code into functions to make the main code simpler
def load_pickle_data (data_dir, noisy_grating_analysis, stimulusFolder):

    datasets_to_load = []
    _to_load = os.listdir(data_dir)
    if noisy_grating_analysis:
        _variable = 'SNR'
        _variable = 'noise std'
    else:
        _variable = 'Luminance'

    for d in _to_load:
        if d.endswith(".pickle"):
            datasets_to_load.append(d)
            
                        
    properties = ['SNR','Reliab','depth','slope']
    combined_df = pd.DataFrame(columns=properties)
    all_rois = []
    all_traces=[]
    tunings = []
    baselines = []
    baseline_power = []
    z_tunings = []

    for idataset, dataset in enumerate(datasets_to_load):
        if not(dataset.split('.')[-1] =='pickle'):
            warnings.warn('Skipping non pickle file: {f}\n'.format(f=dataset))
            continue
        load_path = os.path.join(data_dir, dataset)
        infile = open(load_path, 'rb')
        try:
            workspace = cPickle.load(infile)
        except ImportError:
            print('Unable to import skipping: \n{f} '.format(f=load_path))
            continue
        curr_rois = workspace['final_rois']
        
        stimulus = '{}.txt'
        stimulus = stimulus.format(stimulusFolder)
        if(not(stimulus in curr_rois[0].stim_name)):
            continue
        
        # Thresholding
        # Reliability thresholding
        curr_rois = ROI_mod.analyze_gratings_1Hz(curr_rois)
        curr_rois = ROI_mod.find_inverted(curr_rois,stim_type = '1Hz_gratings')
        curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'reliability':0.8, 'inverted':('b',0.5)})

        if len(curr_rois) <= 1:
            warnings.warn('{d} contains 1 or no ROIs, skipping'.format(d=dataset))
            continue
        
        # get some parameters
        geno, all_rois, roi_data = get_roi_geno (curr_rois, all_rois)
        tunings, z_tunings, baselines, baseline_power = get_turning_power_and_baseline(curr_rois, tunings, z_tunings, baselines, baseline_power)
        print(curr_rois[0].experiment_info['Genotype'])
        print(curr_rois[0].stim_name)
        
        rois_df = get_roi_df (curr_rois, roi_data, geno)
        combined_df = combined_df.append(rois_df, ignore_index=True, sort=False)
        print('{ds} successfully loaded\n'.format(ds=dataset))

    return all_rois, combined_df, tunings, z_tunings, baselines, baseline_power, _variable

#%%

def get_roi_geno (curr_rois, all_rois):

    geno = curr_rois[0].experiment_info['Genotype'][:3]
    if 'ExpLine' in curr_rois[0].experiment_info['Genotype']:
        geno = geno + '_ExpLine'
    elif 'PosCnt' in curr_rois[0].experiment_info['Genotype']:
        geno = geno + '_PosCnt'
    
    all_rois.append(curr_rois)
    data_to_extract = ['SNR','reliability','slope','category','base_slope']
    roi_data = ROI_mod.data_to_list(curr_rois, data_to_extract)

    return geno, all_rois, roi_data 

def get_turning_power_and_baseline(curr_rois, tunings, z_tunings, baselines, baseline_power ):

    curr_tuning = np.squeeze\
                   (list(map(lambda roi: roi.power_at_hz,curr_rois)))
    tunings.append(curr_tuning)
    z_tunings.append(stats.zscore(curr_tuning,axis=1))
    
    curr_base = np.squeeze\
                   (list(map(lambda roi: roi.baselines,curr_rois)))
    baselines.append(curr_base)

    curr_baseP = np.squeeze\
                   (list(map(lambda roi: roi.base_power,curr_rois)))
    baseline_power.append(curr_baseP)

    return tunings, z_tunings, baselines, baseline_power

def get_roi_df (curr_rois, roi_data, geno):

    depths = list(map(lambda roi : roi.imaging_info['depth'], curr_rois))
    df_c = {}
    df_c['depth'] = depths
    
    if "RF_map_norm" in curr_rois[0].__dict__.keys():
        df_c['RF_map_center'] = list(map(lambda roi : (roi.RF_map_norm>0.95).astype(int)
                                             , curr_rois))
        df_c['RF_map_bool'] = np.tile(True,len(curr_rois))
        screen = np.zeros(np.shape(curr_rois[0].RF_map))
        screen[np.isnan(curr_rois[0].RF_map)] = -0.1
        
        for roi in curr_rois:
            curr_map = (roi.RF_map_norm>0.95).astype(int)
    
            x1,x2 = ndimage.measurements.center_of_mass(curr_map)
            s1,s2 = ndimage.measurements.center_of_mass(np.ones(shape=screen.shape))
            roi.distance_to_center = np.sqrt(np.square(x1-s1) + np.square(x2-s2))
            roi.horizontal_pos_screen = x2
            roi.vertical_pos_screen = x1
        df_c['RF_distance_to_center'] = list(map(lambda roi : roi.distance_to_center, 
                                             curr_rois))
        df_c['RF_horizontal_center'] = list(map(lambda roi : roi.horizontal_pos_screen, 
                                             curr_rois))
        df_c['RF_vertical_center'] = list(map(lambda roi : roi.vertical_pos_screen, 
                                             curr_rois))
        print('RFs found')
    else:
        df_c['RF_map_center'] = np.tile(None,len(curr_rois))
        df_c['RF_map_bool'] = np.tile(False,len(curr_rois))
        df_c['RF_distance_to_center'] = np.tile(np.nan,len(curr_rois))
        df_c['RF_horizontal_center'] = np.tile(np.nan,len(curr_rois))
        df_c['RF_vertical_center'] = np.tile(np.nan,len(curr_rois))
        
    
    df_c['SNR'] = roi_data['SNR']
    df_c['slope'] = roi_data['slope']
    df_c['base_slope'] = roi_data['base_slope']
    df_c['category'] = roi_data['category']
    df_c['reliability'] = roi_data['reliability']
    df_c['flyID'] = np.tile(curr_rois[0].experiment_info['FlyID'],len(curr_rois))
    df_c['Geno'] = np.tile(geno,len(curr_rois))
    df_c['uniq_id'] = np.array(map(lambda roi : roi.uniq_id, curr_rois)) 
    df = pd.DataFrame.from_dict(df_c) 
    rois_df = pd.DataFrame.from_dict(df)
    
    return rois_df

#%%
def concatinate_flies (all_rois, combined_df, tunings, z_tunings, baselines, baseline_power):
    
    le = preprocessing.LabelEncoder()
    le.fit(combined_df['flyID'])
    combined_df['flyIDNum'] = le.transform(combined_df['flyID'])

    all_rois=np.concatenate(all_rois)
    tunings = np.concatenate(tunings)
    z_tunings = np.concatenate(z_tunings)
    baselines = np.concatenate(baselines)
    baseline_power = np.concatenate(baseline_power)
    
    return all_rois, combined_df, tunings, z_tunings, baselines, baseline_power

#%%
def plot_only_cat_params(plot_only_cat, cat_dict, combined_df, geno, polarity_dict):
    if plot_only_cat:
        try:
            cat_dict[geno[0:3]]
            curr_neuron_mask = ((combined_df['Geno']==geno) & \
                (combined_df['category']==cat_dict[geno]) & \
                    (combined_df['stim_type']==polarity_dict[geno[0:3]]))
        except KeyError:
            print('No pre-defined category for {g} found. Taking all...\n'.format(g=geno))
            curr_neuron_mask = (combined_df['Geno']==geno)
    else:
        curr_neuron_mask = ((combined_df['Geno']==geno) &\
                            (combined_df['stim_type']==polarity_dict[geno[0:3]]))
    
    return curr_neuron_mask
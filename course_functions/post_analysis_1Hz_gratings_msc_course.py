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
#code_path = r'\\fs02\smolina$\Dokumente\GitHub\scripts\python_scripts\2p_analysis\Helpers' # It must be change for the path where the code is in each PC
#code_path2 = r'\\fs02\smolina$\Dokumente\GitHub\scripts\python_scripts\2p_analysis\Post_analysis'
sys.path.insert(0, code_path)
#sys.path.insert(0, code_path2)  
os.chdir(code_path)

import ROI_mod_reduced_msc_course as ROI_mod
import post_analysis_core as pac
import post_analysis_1Hz_gratings_core_msc_course as pag


#%%
#noisy_grating_analysis = True
#experiment = 'Mi1_GluCla_Mi1_suff_exp'
#stimulusFolder = 'Gratings_sine_white_noise_30sw_30deg_sec_1hz_5sec_DARK_5sec_moving_2_to_0.35_80sec'
#dataFolder = r'F:\SebastianFilesExternalDrive\Science\PhDAGSilies\2pData Python_data'
#initialDirectory = os.path.join(dataFolder, experiment)
#analyzed_data_dir = os.path.join(initialDirectory, 'analyzed_data')
#all_data_dir = os.path.join(analyzed_data_dir , stimulusFolder)
#exp_folder = 'Exp_Cnt'
#data_dir = os.path.join(all_data_dir,exp_folder)
#results_save_dir = os.path.join(data_dir ,'results')
#if not os.path.exists(results_save_dir):
#        os.mkdir(results_save_dir)
#std_signal = 0.17469696946961763

#%%
noisy_grating_analysis = False
experiment = 'Mi1_GluCla_Mi1_suff_exp'
stimulusFolder = 'Gratings_sine_wave_30sw_30deg_sec_1hz_4sec_static_DARK_4sec_moving_5_luminances_0.1_0.5_40sec'
#dataFolder = r'F:\SebastianFilesExternalDrive\Science\PhDAGSilies\2pData Python_data'
dataFolder = r'\\fs02\jcornean$\Dokumente\python_seb_lumgrating_course\2p_analysis\Data_Sebastian'
initialDirectory = os.path.join(dataFolder, experiment)
analyzed_data_dir = os.path.join(initialDirectory, 'analyzed_data')
all_data_dir = os.path.join(analyzed_data_dir , stimulusFolder)
exp_folder = 'PosCnt'  # 'Exp_Cnt'  'PosCnt' 'ExpLine' 
data_dir = os.path.join(all_data_dir,exp_folder)
results_save_dir = os.path.join(data_dir ,'results')
if not os.path.exists(results_save_dir):
    os.mkdir(results_save_dir)




#%%

plot_only_cat = True
polarity_dict, cat_dict = pag.get_polarity_and_cat_dict()
all_rois, combined_df, tunings, z_tunings, baselines, baseline_power, _variable = pag.load_pickle_data (data_dir, noisy_grating_analysis, stimulusFolder)
all_rois, combined_df, tunings, z_tunings, baselines, baseline_power = pag.concatinate_flies (all_rois, combined_df, tunings, z_tunings, baselines, baseline_power)


#%%
_, colors = pac.run_matplotlib_params()

c_dict = {k:colors[k] for k in colors if k in combined_df['Geno'].unique()}

#%% Plotting
## Common attributes for the plots
plt.rcParams['axes.spines.top'] = False #Removing lines of the box around all plots
plt.rcParams['axes.spines.right'] = False #Removing lines of the box around all plots
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.labelsize']=12


#%% Plot single traces
plt.close('all')
for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    # if geno != 'L3_':
    #     continue
    geno_color = c_dict[geno] #  colors[genotype_colors[0]]
    # neuron_save_dir = os.path.join(results_save_dir,geno,'luminance_edges')
    save_n = '{g}_raw'.format(g=geno)
    neuron_save_dir = os.path.join(results_save_dir,save_n)
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    
    curr_neuron_mask = pag.plot_only_cat_params(plot_only_cat, cat_dict, combined_df, geno, polarity_dict)

    curr_rois = all_rois[curr_neuron_mask]
    for idx, roi in enumerate(curr_rois):
        # fig = plt.figure(figsize=(16, 3))
        # roi.plotDF(color=geno_color,line_w=2)
        # save_name = '{geno}ROI_{n}_100p'.format(geno=geno,n=idx)
        # os.chdir(neuron_save_dir)
        # plt.savefig('%s.png' % save_name, bbox_inches='tight')
        # plt.close('all')
        
        fig = plt.figure(figsize=(16, 3))
        plt.plot(roi.conc_resp,color = geno_color)
        save_name = '{geno}_ROI_{n}'.format(geno=geno,n=idx)
        plt.title('baseline_slope: {bs}'.format(bs=roi.base_slope))
        os.chdir(neuron_save_dir)
        plt.savefig('%s.png' % save_name, bbox_inches='tight')
        plt.savefig('%s.pdf' % save_name, bbox_inches='tight')
        plt.close('all')
#%%
ax2 = plt.axes()
sns.boxplot(x="Geno", y="base_slope", data=combined_df, palette=colors)

ax2.set_xlabel('Neuron')
ax2.set_ylabel('Base slope (mean resp)')
# ax2.set_ylim((-3,4))
#%%
combined_df

#%% Print the results
plt.close('all')

for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    geno_color = c_dict[geno] #  colors[genotype_colors[0]]
    # neuron_save_dir = os.path.join(results_save_dir,geno,'luminance_edges')
    neuron_save_dir = os.path.join(results_save_dir,'sine1Hz')
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    
    curr_neuron_mask = pag.plot_only_cat_params(plot_only_cat, cat_dict, combined_df, geno, polarity_dict)
    curr_df = combined_df[curr_neuron_mask]
    
    diff_luminances = all_rois[curr_neuron_mask][0].luminances
    if noisy_grating_analysis:
        std_signal_list = len(diff_luminances)* [std_signal]
        diff_luminances = all_rois[curr_neuron_mask][0].stim_info['SNR'][1:] # Seb added
        diff_luminances = [i / j for i, j in zip(std_signal_list, all_rois[curr_neuron_mask][0].stim_info['SNR'][1:])] # Seb added new
    

    # AX0
    sensitivities = tunings[curr_neuron_mask]
    properties = ['Luminance', 'Response']
    senst_df = pd.DataFrame(columns=properties)
    
    
    tuning_curves = baselines[curr_neuron_mask]
    
    a=pac.compute_over_samples_groups(data = tuning_curves, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyIDNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))


    all_mean_data = a['experiment_ids'][geno]['over_groups_mean']
    print(geno,all_mean_data)
    
    
#%%1 Hz
plt.close('all')

for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    geno_color = c_dict[geno] #  colors[genotype_colors[0]]
    # neuron_save_dir = os.path.join(results_save_dir,geno,'luminance_edges')
    neuron_save_dir = os.path.join(results_save_dir,'sine1Hz')
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    
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
            
    curr_df = combined_df[curr_neuron_mask]
    fig = plt.figure(figsize=(4, 4))
    grid = plt.GridSpec(1, 1, wspace=0.3, hspace=1)
    
    ax1=plt.subplot(grid[0,0])

    
    diff_luminances = all_rois[curr_neuron_mask][0].luminances
    if noisy_grating_analysis:
        diff_luminances = all_rois[curr_neuron_mask][0].stim_info['SNR'][1:] # Seb added
        diff_luminances = [i / j for i, j in zip(std_signal_list, all_rois[curr_neuron_mask][0].stim_info['SNR'][1:])]

    
    cmap = matplotlib.cm.get_cmap('inferno')
    norm = matplotlib.colors.Normalize(vmin=0, 
                                       vmax=np.max(diff_luminances))
    
    # AX0
    sensitivities = tunings[curr_neuron_mask]
    properties = ['Luminance', 'Response']
    senst_df = pd.DataFrame(columns=properties)
    
    
    tuning_curves = tunings[curr_neuron_mask]
    
    a=pac.compute_over_samples_groups(data = tuning_curves, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyIDNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

    label = '{g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))
    all_mean_data = a['experiment_ids'][geno]['over_groups_mean']
    print(geno,all_mean_data)
    all_yerr = a['experiment_ids'][geno]['over_groups_error']
    ax1.errorbar(diff_luminances,all_mean_data,all_yerr,
                 fmt='-s',alpha=1,color=geno_color,label=label)
    ax1.set_ylim((0,ax1.get_ylim()[1]))
    ax1.set_title('1Hz response')
    
    # Saving figure
    save_name = '_1Hzresp_{geno}'.format(geno=geno)
    os.chdir(neuron_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
#%% Plotting Summary per each Genotype
plt.close('all')

for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    geno_color = c_dict[geno] #  colors[genotype_colors[0]]
    # neuron_save_dir = os.path.join(results_save_dir,geno,'luminance_edges')
    neuron_save_dir = os.path.join(results_save_dir,'sine1Hz')
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    
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
            
    curr_df = combined_df[curr_neuron_mask]
    fig = plt.figure(figsize=(8, 3))
    grid = plt.GridSpec(1, 3, wspace=0.3, hspace=1)
    
    ax1=plt.subplot(grid[0,0])
    ax2=plt.subplot(grid[0,1])
    ax3=plt.subplot(grid[0,2])
    
    diff_luminances = all_rois[curr_neuron_mask][0].luminances
    if noisy_grating_analysis:
        diff_luminances = all_rois[curr_neuron_mask][0].stim_info['SNR'][1:] # Seb added
        diff_luminances = [i / j for i, j in zip(std_signal_list, all_rois[curr_neuron_mask][0].stim_info['SNR'][1:])]
    
    cmap = matplotlib.cm.get_cmap('inferno')
    norm = matplotlib.colors.Normalize(vmin=0, 
                                       vmax=np.max(diff_luminances))
    
    # AX0
    sensitivities = tunings[curr_neuron_mask]
    properties = ['Luminance', 'Response']
    senst_df = pd.DataFrame(columns=properties)
    
    
    tuning_curves = tunings[curr_neuron_mask]
    
    a=pac.compute_over_samples_groups(data = tuning_curves, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyIDNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

    label = '{g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))
    all_mean_data = a['experiment_ids'][geno]['over_groups_mean']
    print(geno,all_mean_data)
    all_yerr = a['experiment_ids'][geno]['over_groups_error']
    ax1.errorbar(diff_luminances,all_mean_data,all_yerr,
                 fmt='-s',alpha=1,color=geno_color,label=label)
    ax1.set_ylim((0,ax1.get_ylim()[1]))
    ax1.set_title('1Hz response')
    ax1.set_xlabel(_variable)
    ax1.set_ylabel('Power(a.u.)')

    # AX1
     
    bases = baselines[curr_neuron_mask]
    
    a=pac.compute_over_samples_groups(data = bases, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyIDNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

    label = '{g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))
    all_mean_data = a['experiment_ids'][geno]['over_groups_mean']
    all_yerr = a['experiment_ids'][geno]['over_groups_error']
    ax2.errorbar(diff_luminances,all_mean_data,all_yerr,
                 fmt='-s',alpha=1,color=geno_color,label=label)
    ax2.set_title('Mean response')
    ax2.set_ylim([0,ax2.get_ylim()[1]])
    ax2.set_xlabel(_variable)
    ax2.set_ylabel('dF/F')
    
    
    bins_list = np.linspace(-0.3,0.3,30)
    plt.hist(curr_df['slope'],range=(0,0.2),color=geno_color,bins=bins_list)
    
    ax3.set_title('1Hz response')   
    ax3.set_xlabel('Slope')
    ax3.set_ylabel('Counts')
    # Saving figure
    save_name = '_Summary_{geno}'.format(geno=geno)
    os.chdir(neuron_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
    
    # Tunings
    plt.figure(figsize=(1,6))
    plot_sens = (sensitivities/sensitivities.max(axis=1).reshape(sensitivities.shape[0],1))
    plt.imshow(plot_sens,vmin=0,vmax=1)
    plt.colorbar()
    save_name = 'Tunings_1Hz_{geno}'.format(geno=geno)
    os.chdir(neuron_save_dir)
    #plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
    
    
    # Sorted tunings
    plt.figure(figsize=(1,6))
    maxs = np.argmax(plot_sens,axis=1)
    sorted_indices = np.argsort(maxs)
    sorted_sens = plot_sens[sorted_indices,:]
    
    plt.imshow(sorted_sens,vmin=0,vmax=1)
    plt.colorbar()
    save_name = 'Sorted_tunings_1Hz_{geno}'.format(geno=geno)
    os.chdir(neuron_save_dir)
    #plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
    


#%% Plot them together
plt.close('all')
fig = plt.figure(figsize=(8, 3))
grid = plt.GridSpec(2,3, wspace=0.3, hspace=1)

ax1=plt.subplot(grid[0,:])
ax2=plt.subplot(grid[1,:])

bar_idx =0
for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    if not(geno[:3] in ['L3_','Tm1','Tm2','Tm4','Tm9','Mi1']):
        continue
    bar_idx += 1
    geno_color = c_dict[geno] #  colors[genotype_colors[0]]
    # neuron_save_dir = os.path.join(results_save_dir,geno,'luminance_edges')
    neuron_save_dir = os.path.join(results_save_dir,'sine1Hz')
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    
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
            
    curr_df = combined_df[curr_neuron_mask]
    
    
    
    if geno[:3] in ['Tm1','Tm2','Tm4','Tm9','Mi1']:
        label = '{}- 1Hz response slope'.format(geno)
        # plot_data = stats.zscore(curr_df['slope']) 
        plot_data = curr_df['slope']
        sns.distplot(plot_data,ax=ax1,kde=True,
                     kde_kws={"alpha": .7,'cumulative': False,"lw": 3},
                     color=(geno_color[0],geno_color[1],geno_color[2],1.0),
                         hist=False,bins=10,label=label)

        label = '{}- mean response slope'.format(geno)
        # plot_data = stats.zscore(curr_df['base_slope']) 
        plot_data = curr_df['base_slope']
        sns.distplot(plot_data,ax=ax2,kde=True,
                     kde_kws={"alpha": .7,'cumulative': False, "lw": 3,},
                     color=(geno_color[0],geno_color[1],geno_color[2],1.0),
                         hist=False,bins=10,label=label)
    

    
ax1.set_title('1 Hz power')
ax1.plot([0,0],[0,ax1.get_ylim()[1]],'--k',linewidth=1)
# ax1.set_ylim([0,ax1.get_ylim()[1]-0.4])
ax1.set_xlabel('Slope')
ax2.set_title('Mean')
ax2.plot([0,0],[0,ax2.get_ylim()[1]],'--k',linewidth=1)
# ax2.set_ylim([0,ax2.get_ylim()[1]-0.4])

#save_name = '1hz_and_mean_slopes'
#os.chdir(neuron_save_dir)
#plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

#%% Plot for different flies
plt.close('all')


for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    if not(geno[:3] in ['L2_','L3_','Tm9','Tm1','Mi1']):
        continue
    
    fig = plt.figure(figsize=(4, 4))
    
    
    curr_color =  c_dict[geno] # colors[genotype_colors[idx]]
    # neuron_save_dir = os.path.join(results_save_dir,geno,'luminance_edges')
    neuron_save_dir = os.path.join(results_save_dir,'sine1Hz')
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    
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
            
    curr_df = combined_df[curr_neuron_mask]
    
    sns.scatterplot('base_slope','slope',data=curr_df,hue='flyIDNum',
                    s= 30, alpha=.8,palette='Set2')
    plt.title(geno)
    ca = plt.gca()
    plt.plot([0,0],[ca.get_ylim()[0],ca.get_ylim()[1]],'--k')
    ca.set_ylim([ca.get_ylim()[0],ca.get_ylim()[1]])
    max_x = np.max(ca.get_xlim())
    plt.xlim([-max_x,max_x])
    plt.plot([ca.get_xlim()[0],ca.get_xlim()[1]],[0,0],'--k')
    ca.set_xlim([ca.get_xlim()[0],ca.get_xlim()[1]])
    
#%% Plot tunings for flies
plt.close('all')


for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    if not(geno[:3] in ['L2_','L3_','Tm9','Tm1','Mi1']):
        continue
    
    fig = plt.figure(figsize=(4, 14))
    fig.suptitle('{g}'.format(g=geno), fontsize=16)
    grid = plt.GridSpec(1, 4, wspace=0.3, hspace=0.1)
    
    ax1=plt.subplot(grid[0,0])
    ax2=plt.subplot(grid[0,1])
    ax3=plt.subplot(grid[0,2])
    ax4=plt.subplot(grid[0,3])
    
    
    curr_color =  c_dict[geno] # colors[genotype_colors[idx]]
    # neuron_save_dir = os.path.join(results_save_dir,geno,'luminance_edges')
    neuron_save_dir = os.path.join(results_save_dir,'sine1Hz')
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    
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
            
    curr_df = combined_df[curr_neuron_mask]
    
    mean_baselines = baselines[curr_neuron_mask]
    
    # Baseline tunings
    plot_sens = (mean_baselines/mean_baselines.max(axis=1).reshape(mean_baselines.shape[0],1))
    ax1.imshow(plot_sens,vmin=0,vmax=1)
    ax1.set_title('Mean responses')
    # ax1.colorbar()
    # save_name = 'All_tunings_Sine1Hz_{geno}'.format(geno=geno)
    # os.chdir(neuron_save_dir)
    # plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
    
    
    # Sorted tunings
    maxs = np.argmax(plot_sens,axis=1)
    sorted_indices = np.argsort(maxs)
    sorted_sens = plot_sens[sorted_indices,:]
    
    ax2.imshow(sorted_sens,vmin=0,vmax=1)
    # ax2.set_title('Mean responses sorted')
    # ax2.colorbar()
    # save_name = 'Sorted_tunings_Sine1Hz_{geno}'.format(geno=geno)
    # os.chdir(neuron_save_dir)
    # plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
    
    tunings_1Hz = tunings[curr_neuron_mask]
    # 1Hz tunings
    plot_sens = (tunings_1Hz/tunings_1Hz.max(axis=1).reshape(tunings_1Hz.shape[0],1))
    ax3.imshow(plot_sens,vmin=0,vmax=1)
    # ax3.set_title('1 Hz responses')
    # ax1.colorbar()
    # save_name = 'All_tunings_Sine1Hz_{geno}'.format(geno=geno)
    # os.chdir(neuron_save_dir)
    # plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
    
    
    # Sorted tunings
    maxs = np.argmax(plot_sens,axis=1)
    sorted_indices = np.argsort(maxs)
    sorted_sens = plot_sens[sorted_indices,:]
    
    ax4.imshow(sorted_sens,vmin=0,vmax=1)
    ax4.set_title('1 Hz responses sorted')
    # ax2.colorbar()
    # save_name = 'Sorted_tunings_Sine1Hz_{geno}'.format(geno=geno)
    # os.chdir(neuron_save_dir)
    # plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
    
#%% Heat plots for mean response per each genotype
plt.close('all')


for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    if not(geno[:3] in ['L2_','L3_','Tm9','Tm1','Mi1']):
        continue
    
    curr_color =  c_dict[geno] # colors[genotype_colors[idx]]
    # neuron_save_dir = os.path.join(results_save_dir,geno,'luminance_edges')
    neuron_save_dir = os.path.join(results_save_dir)
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    
    if plot_only_cat:
        try:
            cat_dict[geno]
            curr_neuron_mask = ((combined_df['Geno']==geno) & \
                (combined_df['category']==cat_dict[geno]) & \
                    (combined_df['stim_type']==polarity_dict[geno]))
        except KeyError:
            print('No pre-defined category for {g} found. Taking all...\n'.format(g=geno))
            curr_neuron_mask = (combined_df['Geno']==geno)
    else:
        curr_neuron_mask = ((combined_df['Geno']==geno) &\
                            (combined_df['stim_type']==polarity_dict[geno]))
            
    curr_df = combined_df[curr_neuron_mask]
    
    mean_baselines = baselines[curr_neuron_mask]
    
    fig = plt.figure(figsize=(4, 6))
    # Baseline tunings
    plot_sens = (mean_baselines/mean_baselines.max(axis=1).reshape(mean_baselines.shape[0],1))
    sns.heatmap(plot_sens,cmap='Reds',vmin=0,vmax=1)
    plt.title('Mean responses {g}'.format(g=geno))
#%% Plot all together: 1hz responses and Mean/Baseline response
    
plt.close('all')
fig = plt.figure(figsize=(8, 3))
grid = plt.GridSpec(1,2, wspace=0.3, hspace=1)

ax1=plt.subplot(grid[0,0])
ax2=plt.subplot(grid[0,1])

for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    geno_color = c_dict[geno] # colors[genotype_colors[idx]]
    # neuron_save_dir = os.path.join(results_save_dir,geno,'luminance_edges')
    neuron_save_dir = os.path.join(results_save_dir,'sine1Hz')
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    
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
            
    curr_df = combined_df[curr_neuron_mask]
    
    
    diff_luminances = all_rois[curr_neuron_mask][0].luminances
    if noisy_grating_analysis:
        diff_luminances = all_rois[curr_neuron_mask][0].stim_info['SNR'][1:] # Seb added
        diff_luminances = [i / j for i, j in zip(std_signal_list, all_rois[curr_neuron_mask][0].stim_info['SNR'][1:])]
    
    cmap = matplotlib.cm.get_cmap('inferno')
    norm = matplotlib.colors.Normalize(vmin=0, 
                                       vmax=np.max(diff_luminances))
    
    # AX1
    tuning_curves = tunings[curr_neuron_mask]
    
    a=pac.compute_over_samples_groups(data = tuning_curves, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyIDNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

    label = '{g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))
    
    mean_flies = np.array(a['experiment_ids'][geno]['over_samples_means'])
    #mean_flies= mean_flies/mean_flies.max(1).reshape(mean_flies.shape[0],1) # For normalization, uncomment
    all_mean_data = mean_flies.mean(0)
    all_yerr = mean_flies.std(0)/np.sqrt(mean_flies.shape[0])
    ax1.errorbar(diff_luminances,all_mean_data,all_yerr,
                 fmt='-o',alpha=.8,color=geno_color,label=label)
    
    # AX2
    tuning_curves = baselines[curr_neuron_mask]
    
    a=pac.compute_over_samples_groups(data = tuning_curves, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyIDNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

    label = '{g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))
    
    mean_flies = np.array(a['experiment_ids'][geno]['over_samples_means'])
    # mean_flies= mean_flies/mean_flies.max(1).reshape(mean_flies.shape[0],1)  # For normalization, uncomment
    all_mean_data = mean_flies.mean(0)
    all_yerr = mean_flies.std(0)/np.sqrt(mean_flies.shape[0])
    ax2.errorbar(diff_luminances,all_mean_data,all_yerr,
                 fmt='-o',alpha=.8,color=geno_color,label=label)
    
ax1.set_ylim((0,ax1.get_ylim()[1]))
ax1.set_title('1Hz response')
ax1.legend(loc=4)
ax1.set_ylabel('Power (a.u.)')
ax1.set_xlabel(_variable)

ax2.set_ylim((0,ax2.get_ylim()[1]))
ax2.set_title('Baseline response')
ax2.legend(loc=4)
ax2.set_ylabel('dF/F')
ax2.set_xlabel(_variable)

#save_name = 'Summary_1hz_mean_resp'.format(geno=geno)
#os.chdir(neuron_save_dir)
#plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
#%% Checking values
for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    geno_color = c_dict[geno] # colors[genotype_colors[idx]]
    # neuron_save_dir = os.path.join(results_save_dir,geno,'luminance_edges')
    neuron_save_dir = os.path.join(results_save_dir,'sine1Hz')
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    
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

    diff_luminances = all_rois[curr_neuron_mask][0].luminances
    if noisy_grating_analysis:
        diff_luminances = all_rois[curr_neuron_mask][0].stim_info['SNR'][1:] # Seb added
        diff_luminances = [i / j for i, j in zip(std_signal_list, all_rois[curr_neuron_mask][0].stim_info['SNR'][1:])]

    tuning_curves = tunings[curr_neuron_mask]
    
    a=pac.compute_over_samples_groups(data = tuning_curves, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyIDNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

    all_mean_data = a['experiment_ids'][geno]['over_groups_mean']
    print('{g} resp 1hz POWER: \n'.format(g=geno))
    print(all_mean_data)

    tuning_curves = baselines[curr_neuron_mask]
    
    a=pac.compute_over_samples_groups(data = tuning_curves, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyIDNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

    all_mean_data = a['experiment_ids'][geno]['over_groups_mean']
    print('{g} resp BASELINE: \n'.format(g=geno))
    print(all_mean_data)

#%% Slope histograms and percentage of ROIs
plt.close('all')
fig = plt.figure(figsize=(8, 3))
grid = plt.GridSpec(1,3, wspace=0.3, hspace=1)

ax1=plt.subplot(grid[0,0])
ax2=plt.subplot(grid[0,1])
ax3=plt.subplot(grid[0,2])

bar_idx =0
ax3_labels=[]
for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    if not(geno[:3] in ['L3_','Tm9','Tm1', 'Mi1']):
        continue
    bar_idx += 1
    geno_color = c_dict[geno] #  colors[genotype_colors[0]]
    # neuron_save_dir = os.path.join(results_save_dir,geno,'luminance_edges')
    neuron_save_dir = os.path.join(results_save_dir,'sine1Hz')
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    
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
            
    curr_df = combined_df[curr_neuron_mask]
    
    
    
    
    #label = '{}- std: {:.3f}'.format(geno,curr_df['slope'].std())
    label = '{}'.format(geno)
    sns.distplot(curr_df['slope'],ax=ax1,kde=True,
                 kde_kws={"alpha": .7,'cumulative': False,"lw": 3},
                 color=(geno_color[0],geno_color[1],geno_color[2],1.0),
                     hist=False,bins=10,label=label)
    
    #label = '{}- std: {:.3f}'.format(geno,curr_df['base_slope'].std())
    label = '{}'.format(geno)
    sns.distplot(curr_df['base_slope'],ax=ax2,kde=True,
                 kde_kws={"alpha": .7,'cumulative': False, "lw": 3},
                 color=(geno_color[0],geno_color[1],geno_color[2],1.0),
                     hist=False,bins=10,label=label)
    
    pos = np.where(curr_df['base_slope']>0)[0].shape[0]/float(curr_df['base_slope'].shape[0])
    neg = np.where(curr_df['base_slope']<0)[0].shape[0]/float(curr_df['base_slope'].shape[0])
    ax3.bar(bar_idx,pos,color='y')
    ax3.bar(bar_idx,-neg,color='k')
    ax3_labels.append(geno)
    
ax1.set_title('1 Hz power')
ax1.plot([0,0],[0,ax1.get_ylim()[1]],'--k',linewidth=1)
ax1.set_ylim([0,ax1.get_ylim()[1]-0.4])
ax1.set_xlabel('Slope')

ax2.set_title('Mean')
ax2.plot([0,0],[0,ax2.get_ylim()[1]],'--k',linewidth=1)
ax1.set_ylim([0,ax1.get_ylim()[1]-0.4])
ax2.set_xlabel('Slope')

ax3.set_title('+ and - mean response slopes')
ax3.set_ylabel('Percentage of ROIs')
ax3.set_ylim([-1,1])

ax3.set_xlabel(ax3_labels)

save_name = 'Summary_slopes'
os.chdir(neuron_save_dir)
plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)  
#%% Stats
plt.close('all')

for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    geno_color = c_dict[geno] # colors[genotype_colors[idx]]
    # neuron_save_dir = os.path.join(results_save_dir,geno,'luminance_edges')
    neuron_save_dir = os.path.join(results_save_dir,'sine1Hz')
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    
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
            
    curr_df = combined_df[curr_neuron_mask]
    
    diff_luminances = all_rois[curr_neuron_mask][0].luminances
    if noisy_grating_analysis:
        diff_luminances = all_rois[curr_neuron_mask][0].stim_info['SNR'][1:] # Seb added
        diff_luminances = [i / j for i, j in zip(std_signal_list, all_rois[curr_neuron_mask][0].stim_info['SNR'][1:])]
    
    
    # AX1
     
    #bases = baselines[curr_neuron_mask]
    bases = tunings[curr_neuron_mask]
    a=pac.compute_over_samples_groups(data = bases, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyIDNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))
    
    b = np.array(a['experiment_ids'][geno]['over_samples_means'])
    stats_df = pd.DataFrame(data=b,columns = diff_luminances) 
    import scipy.stats as stats
    import statsmodels.stats.multicomp as mc
    
    fvalue, pvalue = stats.f_oneway(stats_df[diff_luminances[0]],
                                    stats_df[diff_luminances[1]],
                                    stats_df[diff_luminances[2]],
                                    stats_df[diff_luminances[3]],
                                    stats_df[diff_luminances[4]])
    
    print(fvalue, pvalue)
    
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    # reshape the d dataframe suitable for statsmodels package 
    d_melt = pd.melt(stats_df.reset_index(), id_vars=['index'], value_vars=[diff_luminances[0],
                                                                     diff_luminances[1], 
                                                                     diff_luminances[2],
                                                                     diff_luminances[3],
                                                                     diff_luminances[4]])
    # replace column names
    d_melt.columns = ['index', 'luminances', 'value']
    # Ordinary Least Squares (OLS) model
    model = ols('value ~ C(luminances)', data=d_melt).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table
    
    import statsmodels.stats.multicomp as mc

    comp = mc.MultiComparison(d_melt['value'], d_melt['luminances'])
    post_hoc_res = comp.tukeyhsd()
    post_hoc_res.summary()
    print(post_hoc_res.summary())
    post_hoc_res.plot_simultaneous(ylabel= "Luminances", xlabel= "Score Difference")
    
    #
    

    # comp = mc.MultiComparison(df['libido'], df['dose'])
    # post_hoc_res = comp.tukeyhsd()
    # post_hoc_res.summary()
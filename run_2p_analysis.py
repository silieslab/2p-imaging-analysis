 #-*- coding: utf-8 -*-

"""
Created on Thu July 15 08:29:00 2022

@author: Sebastian Molina-Obando

Run script for the main analysis of calcium imaging data
"""
#Importing analysis function
from main_analysis.main_2p_imaging_analysis import main_2p_imaging_analysis
import pandas as pd
import sys

import tkinter
from tkinter import filedialog


#%% Calling information from a txt file
def data_folder(dataFolder):
    print(dataFolder)
    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    #dataFolder = os.getcwd()
    #dataFolder= data_folfer_path
    #dataFolder = r'D:\Two-Photon-Data-Ultima' # Temporary until chnaging folder name
    print('Please select the meta data file')
    meta_data_path = filedialog.askopenfilename(parent=root, initialdir=dataFolder,
        title=' >>>>>>>>>>>>  Hello dear scientist! From your: <experiment>\\raw-data\\aligned-data folder,   please select the subject and its meta_data file to be analyzed  <<<<<<<<<<<<')
    if len(meta_data_path) > 0:
        print ('--------------------------------------------------------------------------------------------')
        print ("Analysis running for: %s" % meta_data_path)
        print ('--------------------------------------------------------------------------------------------')

    meta_data_df = pd.read_csv(meta_data_path)
    meta_data_dict = meta_data_df.set_index('KEY').T.to_dict('list')
    # Please complete the following information
    analysis_params = {}
    # Fly-specific selection parameters
    #Adding an option to load this information from a metadata file
    analysis_params['experiment'] = meta_data_dict['Experiment'][0]
    analysis_params['current_exp_ID'] = meta_data_dict['Subject_ID'][0]
    analysis_params['current_t_series'] = meta_data_dict['TSeries_ID'][0]#f"TSeries-{meta_data_dict['TSeries'][0]}"
    analysis_params['genotype'] = meta_data_dict['Genotype'][0]
    analysis_params['save_folder_geno'] = meta_data_dict['Condition'][0]
    analysis_params['stim_name'] = meta_data_dict['Stimulus'][0]
    analysis_params['age'] = meta_data_dict['Age'][0]
    analysis_params['sex'] = meta_data_dict['Sex'][0]

    
    # TODO: Move the following options in a pop up window for the user to select from several options
    # Files and analysis type
    analysis_params['time_series_stack'] = f"{meta_data_dict['TSeries_ID'][0]}_Ch2_reg.tif"# f"{analysis_params['current_t_series']}_Ch2_reg.tif" # Name of your motion-aligned tif stack.
    analysis_params['deltaF_method'] = 'mean' # 'mean'
    analysis_params['df_first'] = True # If df_f should be done BEFORE trial averaging. False option has bugs
    analysis_params['int_rate'] = 10 # Rate to interpolate the data
    analysis_params['stimulus_type'] = 'general'
    # 'general' # For general pre analysis of any stimulus. (stimulus agnostic)
    # 'White_Noise' # Specifically for white noise stimulation (this is a temporary option that will be moved to stim-specific postanylsis function)

    # Choose ROI selection/extraction parameters
    analysis_params['roi_extraction_type'] = 'manual' #  'transfer' 'manual' 'cluster_analysis'
    analysis_params['transfer_type'] = 'minimal' # 'minimal' (so far the single option)
    analysis_params['transfer_TSeries'] = 'TSeries-fly4-001' # Choose the other TSeries you want to tranfer the ROIs from
    analysis_params['transfer_data_name'] = f"{analysis_params['current_exp_ID']}-{analysis_params['transfer_TSeries']}_{analysis_params['roi_extraction_type']}.pickle"

    analysis_params['use_avg_data_for_roi_extract'] = False
    analysis_params['use_other_series_roiExtraction'] = False # ROI selection video
    analysis_params['roiExtraction_tseries'] = analysis_params['current_t_series']

    # Saving options
    save_data = True #For saving analyzed data in pickle files AND general plots (non stimulus specific plots)
    #dataFolder = r'D:\2pData Ultima'  # Main folder where the folder for the analysis_params['experiment'] is located

    return analysis_params, dataFolder, save_data 

#For debugging
analysis_params, dataFolder, save_data  = data_folder(r'D:\Two-Photon-Data-Ultima')
main_2p_imaging_analysis(analysis_params, dataFolder, save_data)

#%% Running code from terminal (typing: "python run_2p_analysis.py data_folder D:\Two-Photon-Data-Ultima")
if __name__ == "__main__":
    analysis_params, dataFolder, save_data  =  globals()[sys.argv[1]](sys.argv[2]) # Makes possible to call the data_folder() function in the command line giving a function input
    #analysis_params, dataFolder, save_data  = data_folder()

    #Calling and running the actual analysis function
    main_2p_imaging_analysis(analysis_params, dataFolder, save_data)

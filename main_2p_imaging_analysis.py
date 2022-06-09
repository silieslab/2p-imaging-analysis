# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:27:10 2021

@author: smolina and Burak Gur


GENERAL DISCLAIMER : the code is optimized for "2pstim" stimulation currently running in ULTIMA
For make it compatible for the "C++ stimulation" in the INVESTIGATOR and for "Pystim" in ULTIMA,
two functions in "stim_functions" still need to be implemented: readStimInformation and readStimOut

"""
def main_2p_imaging_analysis(userParams, dataFolder, save_data):

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from core_functions_2p_imaging_analysis import load_movie, get_stim_xml_params,organize_extraction_params,get_epochs_identity, calculate_SNR_Corr,plot_roi_masks,conc_traces,interpolate_signal
    from roi_selection_functions import run_ROI_selection
    from roi_class import generate_ROI_instances, separate_trials_ROI, get_masks_image
    from core_functions_general import saveWorkspace
    from ROI_mod import reverse_correlation_analysis

    #%% Messages to developer

    print('Message to developer: ')
    print('Plots missing for raw data summary. Check:\n################## Plots missing here ##################')

    #%% From the user-defined dictionary of parameters

    # Fly-specific selection parameters
    experiment = userParams['experiment']
    current_exp_ID = userParams['current_exp_ID'] 
    current_t_series = userParams['current_t_series']
    Genotype = userParams['Genotype']
    save_folder_geno = userParams['save_folder_geno']
    Age = userParams['Age']
    Sex = userParams['Sex']


    # ROI selection/extraction parameters
    time_series_stack = userParams['time_series_stack']# 'Raw_stack.tif' 'Mot_corr_stack.tif' # A tif stack.
    roi_extraction_type = userParams['roi_extraction_type'] #  'transfer' 'manual' 'cluster_analysis'
    transfer_type = userParams['transfer_type'] # 'minimal' (so far the single option)
    transfer_data_name = userParams['transfer_data_name']

    use_avg_data_for_roi_extract = userParams['use_avg_data_for_roi_extract'] 
    use_other_series_roiExtraction = userParams['_other_series_roiExtraction'] # ROI selection video
    roiExtraction_tseries = userParams['roiExtraction_tseries']


    deltaF_method = userParams['deltaF_method']  # how F0 is being defined
    df_first = userParams['df_first'] # Ture if df_f should be done BEFORE trial averaging. False option has bugs
    int_rate = userParams['int_rate'] # usually 10 hz extrapolation

    # choose which type of stimulus to analyse (JC)
    stimulus_type = userParams['stimulus_type'] 

    # Saving options
    save_data = save_data

    #%% Auto-setting of some other directories
    dataFolder = dataFolder 
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
    experiment_conditions = {'Genotype' : Genotype, 'Age': Age, 'Sex' : Sex,
                            'FlyID' : current_exp_ID, 'MovieID': current_movie_ID}

    #%% Load of aligned data
    dataDir = os.path.join(alignedDataDir, current_exp_ID, current_t_series)
    time_series = load_movie(dataDir,time_series_stack)
    mean_image = time_series.mean(0)
                                                                                    #imgplot = plt.imshow(mean_image)
                                                                                    #imgplot = plt.imshow(mean_image,cmap="hot")

    #%% Metadata extraction (from xml, stimulus input and stimulus output file) 
    imaging_information,stimulus_information, stimType, rawStimData,stimInputFile = get_stim_xml_params(dataDir, stimInputDir)

    #%% Epochs sorting and identity assignment. Adding info to "stimulus_information"
    stimulus_information = get_epochs_identity(imaging_information,stimulus_information,stimType, rawStimData,stimInputFile)


    #%%  ROI selection

    extraction_params = organize_extraction_params(roi_extraction_type,current_t_series=current_t_series,current_exp_ID=current_exp_ID,
                                alignedDataDir=alignedDataDir,stimInputDir=stimInputDir,use_other_series_roiExtraction = use_other_series_roiExtraction,
                                use_avg_data_for_roi_extract = use_avg_data_for_roi_extract,roiExtraction_tseries=roiExtraction_tseries,
                                transfer_data_n = transfer_data_name,transfer_data_store_dir = saveOutputDir,transfer_type = transfer_type,
                                imaging_information=imaging_information,experiment_conditions=experiment_conditions)
                                #for manual only get the type? yes -JC
        
    ROI_selection_dict = run_ROI_selection(extraction_params,time_series_stack,stimulus_information, imaging_information,image_to_select=mean_image)

    #%%  Creation of ROI objects of a class
    if ROI_selection_dict['rois'] == None:
        del ROI_selection_dict['rois']
        rois = generate_ROI_instances(ROI_selection_dict,
                                            mean_image, 
                                            experiment_info = experiment_conditions, 
                                            imaging_info =imaging_information)

    #%%  Background substraction

    #foreground, background = extract_im_background(time_series, time_series) #JC: changed background substraction method to darkest pixels -2022_02_01
    #background_bool = background.astype(np.bool)    #JC: convert the 1,0 array to Bool (should have worked before but didn't for me, so I changed it)
    # 'time_series = np.transpose(np.subtract(np.transpose(time_series),
    #                                        time_series[:,background_bool].mean(axis=1))) #JC: useing darkest pixel
    time_series = np.transpose(np.subtract(np.transpose(time_series),
                                            time_series[:,ROI_selection_dict['bg_mask']].mean(axis=1))) #JC: using selected bg
    #use bg_mask to substract values from there from the whole file -JC    

    if stimulus_type == '5_sec_FFF':
        # Data sorting (epochs sorting) + Trial averaging (TA) + deltaF/f
        # ROI trial separated responses
        analysis_params = {'deltaF_method': deltaF_method, 'df_first': df_first} 

        (wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs,respTraces_allTrials_ROIs_raw,
        baselineTraces_allTrials_ROIs) = \
            separate_trials_ROI(time_series,rois,stimulus_information,
                                    imaging_information['frame_rate'],moving_avg = True, bins = 3,
                                    df_method = analysis_params['deltaF_method'],df_first = df_first)
        #%%  SNR and reliability
        baseTraces_SNR = baselineTraces_allTrials_ROIs
        if stimulus_information['random'] == 2:
            epoch_to_exclude = None
            baseTraces_SNR = respTraces_allTrials_ROIs_raw.copy()
        elif stimulus_information['random'] == 0:
            epoch_to_exclude = stimulus_information['baseline_epoch']
        else:
            epoch_to_exclude = None

        [SNR_rois, corr_rois] = calculate_SNR_Corr(baseTraces_SNR,
                                                    respTraces_allTrials_ROIs_raw,rois,
                                                    epoch_to_exclude=None)

        #%% Plotting ROIs and properties
        if save_data:
            figure_save_dir = os.path.join(dataDir, 'Results')
        else: 
            figure_save_dir = trash_folder

        if not os.path.exists(figure_save_dir):
            os.mkdir(figure_save_dir)

        roi_image = get_masks_image(rois)
        plot_roi_masks(roi_image,mean_image,len(rois),
                        current_movie_ID,save_fig=True,
                        save_dir=figure_save_dir,alpha=0.4)

        #%% Store relevant information in each roi
        for roi in rois:
            roi.extraction_params = extraction_params
            if roi_extraction_type == 'transfer': # Update transferred ROIs
                roi.experiment_info = experiment_conditions
                roi.imaging_info = imaging_information
                for param in analysis_params.keys():
                    roi.analysis_params[param] = analysis_params[param]
            else:
                roi.analysis_params= analysis_params

        list(map(lambda roi: roi.appendStimInfo(stimulus_information), rois))
        list(map(lambda roi: roi.findMaxResponse_all_epochs(), rois))
        list(map(lambda roi: roi.setSourceImage(mean_image), rois))

        #Data interpolation (to 10 hz)
        print('Seb, split concatenation from interpolation.It does not make sense for some stimuli to concatenate')
        for roi in rois:
            roi.int_whole_trace_all_epochs = roi.whole_trace_all_epochs.copy()
            roi.int_stim_trace = roi.whole_trace_all_epochs.copy()  #difference? -JC
            #JC: whole_trace_all_epochs has mean of trials (df/f) for each frame
            #    for the respective ROI for each epoch stored.
            #    If movingaverage = True, traces were convolved with a certain bin size

            for idx, epoch in enumerate(list(range(0,roi.stim_info['EPOCHS']))): #Seb: epochs_number --> EPOCHS, JC: changed beginning to 0 so first eppch is included
                    stimulus_dur = roi.stim_info['duration'][idx]   #JC: needed for not had coded interpolation
                    roi.int_whole_trace_all_epochs[epoch] = interpolate_signal(roi.int_whole_trace_all_epochs[epoch], 
                                                        roi.imaging_info['frame_rate'], 
                                                        int_rate, 'data', stimulus_dur)
                    roi.int_stim_trace[epoch] = interpolate_signal(roi.int_stim_trace[epoch], 
                                                        roi.imaging_info['frame_rate'], 
                                                        int_rate, 'stim', stimulus_dur)
                    roi.int_rate = int_rate

    #%%  ROI concatenation
        rois = conc_traces(rois, interpolation = True, int_rate = int_rate)

    elif stimulus_type == 'White_Noise':
        #Seb: generating the ternary noise stimulus
        choiseArr = [0,0.5,1]
        x = 16
        y = 16  #JC: changed from 1 to 16 because I have 16X16X10000 stimulus
        z= 10000 # z- dimension (here frames presented over time)
        np.random.seed(54378) #Fix seed. Do not ever change before calling this from stim_output_file
        stim= np.random.choice(choiseArr, size=(z,x,y))

        # ROI raw signals
        for iROI, roi in enumerate(rois):
            roi.raw_trace = time_series[:,roi.mask].mean(axis=1)
            roi.wn_stim = stim

        # Append relevant information and calculate some parameters
        list(map(lambda roi: roi.appendStimInfo(stimulus_information), rois))
        list(map(lambda roi: roi.setSourceImage(mean_image), rois))


        #%% White noise analysis

        rois = reverse_correlation_analysis(rois, noise_type='grit') #JC: added grit option
        #final_rois = rois

    

    #%%  Raw data plot (whole stimulus and response traces for all ROIs, colormaps of masks with SNR-reliability-reponse quality)

    ################## Plots missing here ##################

    #%%  Single Fly summary plot (includes TA trace per epoch, basic response calculations)

    ################## Plots missing here ##################


    #%%  Save data pickle file 
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



    #%% Temporary
    final_rois = rois
    final_roi_image = get_masks_image(final_rois)



                                        
    #%%  Data sorting (epochs sorting) + Trial averaging (TA) + deltaF/f
    # ROI trial separated responses
    analysis_params = {'deltaF_method': deltaF_method, 'df_first': df_first} 

    (wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs,respTraces_allTrials_ROIs_raw,
    baselineTraces_allTrials_ROIs) = \
        separate_trials_ROI(time_series,rois,stimulus_information,
                                imaging_information['frame_rate'],moving_avg = True, bins = 3,
                                df_method = analysis_params['deltaF_method'],df_first = df_first)
    #%%  SNR and reliability
    baseTraces_SNR = baselineTraces_allTrials_ROIs
    if stimulus_information['random'] == 2:
        epoch_to_exclude = None
        baseTraces_SNR = respTraces_allTrials_ROIs_raw.copy()
    elif stimulus_information['random'] == 0:
        epoch_to_exclude = stimulus_information['baseline_epoch']
    else:
        epoch_to_exclude = None

    [SNR_rois, corr_rois] = calculate_SNR_Corr(baseTraces_SNR,
                                                respTraces_allTrials_ROIs_raw,rois,
                                                epoch_to_exclude=None)

    #%% Plotting ROIs and properties
    if save_data:
        figure_save_dir = os.path.join(dataDir, 'Results')
    else: 
        figure_save_dir = trash_folder

    if not os.path.exists(figure_save_dir):
        os.mkdir(figure_save_dir)

    roi_image = get_masks_image(rois)
    plot_roi_masks(roi_image,mean_image,len(rois),
                    current_movie_ID,save_fig=True,
                    save_dir=figure_save_dir,alpha=0.4)

    #%% Store relevant information in each roi
    for roi in rois:
        roi.extraction_params = extraction_params
        if roi_extraction_type == 'transfer': # Update transferred ROIs
            roi.experiment_info = experiment_conditions
            roi.imaging_info = imaging_information
            for param in analysis_params.keys():
                roi.analysis_params[param] = analysis_params[param]
        else:
            roi.analysis_params= analysis_params

    list(map(lambda roi: roi.appendStimInfo(stimulus_information), rois))
    list(map(lambda roi: roi.findMaxResponse_all_epochs(), rois))
    list(map(lambda roi: roi.setSourceImage(mean_image), rois))

    #%%  Data interpolation (to 10 hz)
    print('Seb, whole_trace_all_epoch last too long? Expected 10, received 100')
    for roi in rois:
        roi.int_whole_trace_all_epochs = roi.whole_trace_all_epochs.copy()
        roi.int_stim_trace = roi.whole_trace_all_epochs.copy()

        for idx, epoch in enumerate(list(range(1,roi.stim_info['EPOCHS']))): #Seb: epochs_number --> EPOCHS
                roi.int_whole_trace_all_epochs[epoch] = interpolate_signal(roi.int_whole_trace_all_epochs[epoch], 
                                                    roi.imaging_info['frame_rate'], 
                                                    int_rate)
                curr_stim = np.zeros((1,len(roi.whole_trace_all_epochs[epoch])))[0]
                curr_stim = curr_stim + idx
                roi.int_stim_trace[epoch] = interpolate_signal(curr_stim, 
                                                    roi.imaging_info['frame_rate'], 
                                                    int_rate,int_time = None)
                roi.int_rate = int_rate

    #%%  ROI concatenation
    rois = conc_traces(rois, interpolation = False, int_rate = int_rate)
    #%%  Raw data plot (whole stimulus and response traces for all ROIs, colormaps of masks with SNR-reliability-reponse quality)
    #%%  Single Fly summary plot (includes TA trace per epoch, basic response calculations)
    #%%  Save data pickle file 
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



    #%% Temporary
    final_rois = rois
    final_roi_image = get_masks_image(final_rois)
    return print('2p analysis completed')

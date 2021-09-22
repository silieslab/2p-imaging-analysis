# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:44:39 2021

@author: smolina and Burak Gur
"""
#%% Importing packages

import os
import glob
from skimage import io
import matplotlib.plot as plt
import numpy as np
from xml_functions import getFramePeriod,getMicRelativeTime,getLayerPosition,getPixelSize
from stim_functions import readStimOut, readStimInformation
from epoch_functions import getEpochCount, divideEpochs, divide_all_epochs




#%% Functions
def load_movie (dataDir,stimInputDir,stack):
    

    # Generate necessary directories for figures
    current_t_series=os.path.basename(dataDir)
    
    # Load movie, get stimulus and imaging information
    try: 

        movie_path = os.path.join(dataDir, stack)
        time_series = io.imread(movie_path)
    except IOError:
        movie_path = os.path.join(dataDir, '{t_name}_{stack}'.format(t_name=current_t_series, stack=stack))
        time_series = io.imread(movie_path)
        

    
    return time_series



def get_stim_xml_params(t_series_path, stimInputDir):
    """ Gets the required stimulus and imaging parameters.
    Parameters
    ==========
    t_series_path : str
        Path to the T series folder for retrieving stimulus related information and
        xml file which contains imaging parameters.
    
    stimInputDir : str
        Path to the folder where stimulus input information is located.
        
    Returns
    =======
    stimulus_information : list 
        Stimulus related information is stored here.
        
    imaging_information : XXXXXXXXX
    
    stimType : XXXXXXXXX 
    rawStimData : XXXXXXXXX
    stimInputFile : XXXXXXXXX
        


    """
    
    #%% Metadata from microscope computer xml file
    
    # Finding the xml file and retrieving relevant information
    
    xmlPath = os.path.join(t_series_path, '*-???.xml')
    xmlFile = (glob.glob(xmlPath))[0]
    
    #  Finding the frame period (1/FPS) and layer position
    framePeriod = getFramePeriod(xmlFile=xmlFile)
    frameRate = 1/framePeriod
    layerPosition = getLayerPosition(xmlFile=xmlFile)
    depth = layerPosition[2]
    
    imagetimes = getMicRelativeTime(xmlFile)

    
    # Pixel definitions
    x_size, y_size, pixelArea = getPixelSize(xmlFile)
    
    # Keeping imaging information
    imaging_information = {'frame_rate' : frameRate, 'pixel_size': x_size, 
                             'depth' : depth, 'frame_timings' : imagetimes}
    
    
   #%% Metadata from stimulus computer input and output files
    
    # Stimulus output information
    stimOutPath = os.path.join(t_series_path, '_stimulus_output_*')
    stimOutFile = (glob.glob(stimOutPath))[0]
    (stimType, rawStimData) = readStimOut(stimOutFile=stimOutFile, 
                                          skipHeader=3) # Seb: skipHeader = 3 for _stimulus_ouput from 2pstim
    
    # Stimulus information
    (stimInputFile,stimulus_information) = readStimInformation(stimType=stimType,
                                                      stimInputDir=stimInputDir)
    
    return imaging_information, stimulus_information, stimType, rawStimData,stimInputFile
    
    
    
    
def get_epochs_identity(imaging_information,stimulus_information,stimType, rawStimData,stimInputFile):
    """ Gets specific info about each epoch 
    Parameters
    ==========
   
        
    Returns
    =======
 
    stimulus_information : containing 
                            trialCoor : list of start, end coordinates for each trial for each epoch
                            XXXXXXXXX :
                                ...
    """
    
    isRandom = int(stimulus_information['randomize'][0])
    epochDur = stimulus_information['duration']
    epochDur = [float(sec) for sec in epochDur]
    epochCount = getEpochCount(rawStimData=rawStimData, epochColumn=3)
    framePeriod = 1/imaging_information['frame_rate']
    # Finding epoch coordinates and number of trials, if isRandom is 1 then
    # there is a baseline epoch otherwise there is no baseline epoch even 
    # if isRandom = 2 (which randomizes all epochs)                                        
    if epochCount <= 1:
        trialCoor = 0
        trialCount = 0
    elif isRandom == 1:
        (trialCoor, trialCount, _) = divideEpochs(rawStimData=rawStimData,
                                                 epochCount=epochCount,
                                                 isRandom=isRandom,
                                                 framePeriod=framePeriod ,
                                                 trialDiff=0.20,
                                                 overlappingFrames=0,
                                                 firstEpochIdx=0,
                                                 epochColumn=3,
                                                 imgFrameColumn=7,
                                                 incNextEpoch=True,
                                                 checkLastTrialLen=True)
    else:
        (trialCoor, trialCount) = divide_all_epochs(rawStimData, epochCount, 
                                                    framePeriod, trialDiff=0.20,
                                                    epochColumn=3, imgFrameColumn=7,
                                                    checkLastTrialLen=True)
     

    # Adding more information
    stimulus_information['epoch_dur'] = epochDur # Seb: consider to delete this line. Redundancy
    stimulus_information['random'] = isRandom # Seb: consider to delete this line. Redundancy
    stimulus_information['output_data'] = rawStimData
    stimulus_information['frame_timings'] = imaging_information['frame_timings']
    stimulus_information['stim_name'] = stimType.split('/')[-1]
    stimulus_information['trial_coordinates'] = trialCoor

    if isRandom==0:
        stimulus_information['baseline_epoch'] = 0  
        stimulus_information['baseline_duration'] = \
            stimulus_information['epoch_dur'][stimulus_information['baseline_epoch']]
        stimulus_information['epoch_adjuster'] = 0
        print('\n Stimulus non random, baseline epoch selected as 0th epoch\n')
    elif isRandom == 2:
        stimulus_information['baseline_epoch'] = None
        stimulus_information['baseline_duration'] = None
        stimulus_information['epoch_adjuster'] = 0
        print('\n Stimulus all random, no baseline epoch present\n')
    elif isRandom == 1:
        stimulus_information['baseline_epoch'] = 0 
        stimulus_information['baseline_duration'] = \
            stimulus_information['epoch_dur'][stimulus_information['baseline_epoch']]
        stimulus_information['epoch_adjuster'] = 1
    

        
    return stimulus_information

def organize_extraction_params(extraction_type,
                               current_t_series=None,current_exp_ID=None,
                               alignedDataDir=None,
                               stimInputDir=None,
                               use_other_series_roiExtraction = None,
                               use_avg_data_for_roi_extract = None,
                               roiExtraction_tseries=None,
                               transfer_data_n = None,
                               transfer_data_store_dir = None,
                               transfer_type = None,
                               imaging_information=None,
                               experiment_conditions=None):
    
    """ THIS IS DOING THIS
    Parameters
    ==========
   
    XXXXXXXXXXX
        
    Returns
    =======
 
    XXXXXXXXXXX  
    
    
    """
    
    extraction_params = {}
    extraction_params['type'] = extraction_type
    if extraction_type == 'SIMA-STICA':
        extraction_params['stim_input_path'] = stimInputDir
        if use_other_series_roiExtraction:
            series_used = roiExtraction_tseries
        else:
            series_used = current_t_series
        extraction_params['series_used'] = series_used
        extraction_params['series_path'] = \
            os.path.join(alignedDataDir, current_exp_ID, 
                                  series_used)
        extraction_params['area_max_micron'] = 4
        extraction_params['area_min_micron'] = 1
        extraction_params['cluster_max_1d_size_micron'] = 4
        extraction_params['cluster_min_1d_size_micron'] = 1
        extraction_params['extraction_reliability_threshold'] = 0.4
        extraction_params['use_trial_avg_video'] = \
            use_avg_data_for_roi_extract
    elif extraction_type == 'transfer':
        transfer_data_path = os.path.join(transfer_data_store_dir,
                                          transfer_data_n)
        extraction_params['transfer_data_path'] = transfer_data_path
        extraction_params['transfer_type']=transfer_type
        extraction_params['imaging_information']= imaging_information
        extraction_params['experiment_conditions'] = experiment_conditions
        
        
    return extraction_params


def run_ROI_selection(extraction_params, stack, image_to_select=None):
    """
    THIS IS DOING THIS
    Parameters
    ==========
   
    XXXXXXXXXXX
        
    Returns
    =======
 
    XXXXXXXXXXX  
    

    """
    # Categories can be used to classify ROIs depending on their location
    # Backgroud mask (named "bg") will be used for background subtraction
    plt.close('all')
    plt.style.use("default")
    print('\n\nSelect categories and background')
    [cat_masks, cat_names] = select_regions(image_to_select, 
                                            image_cmap="viridis",
                                            pause_t=8)
    
    # have to do different actions depending on the extraction type
    if extraction_params['type'] == 'manual':
        print('\n\nSelect ROIs')
        [roi_masks, roi_names] = select_regions(image_to_select, 
                                                image_cmap="viridis",
                                                pause_t=4.5,
                                                ask_name=False)
        all_rois_image = generate_roi_masks_image(roi_masks,
                                                  np.shape(image_to_select))
        
        return cat_masks, cat_names, roi_masks, all_rois_image, None, None
            
    elif extraction_params['type'] == 'SIMA-STICA': 
        # Need the time series and information about the video to be extracted
        (time_series, stimulus_information,imaging_information) = \
            pre_processing_movie (extraction_params['series_path'],
                                  extraction_params['stim_input_path'], stack)
        
        # A trial averaged version of the video can be used for extraction 
        # since it may decrease the noise. Yet it can introduce artifacts
        # between the epochs
        if extraction_params['use_trial_avg_video']:
            (avg_video, _, _) = \
                separate_trials_video(time_series,stimulus_information,
                                      imaging_information['frame_rate'])
            sima_dataset = generate_avg_movie(extraction_params['series_path'], 
                                              stimulus_information,
                                              avg_video)
        else:
            movie = np.zeros(shape=(time_series.shape[0],1,time_series.shape[1],
                                time_series.shape[2],1))
            movie[:,0,:,:,0] = time_series
            b = sima.Sequence.create('ndarray',movie)
            sima_dataset = sima.ImagingDataset([b],None)
        
        # We need a certain range of areas for rois 
        area_max_micron = extraction_params['area_max_micron']
        area_min_micron = extraction_params['area_min_micron']
        area_max = int(math.pow(math.sqrt(area_max_micron) / \
                                imaging_information['pixel_size'], 2))
        area_min = int(math.pow(math.sqrt(area_min_micron) / \
                                imaging_information['pixel_size'], 2))
        [roi_masks, all_rois_image] = find_clusters_STICA(sima_dataset,
                                                          area_min,
                                                          area_max)
        threshold_dict = {'SNR': 0.75,'reliability': 0.4}
        
        return cat_masks, cat_names, roi_masks, all_rois_image, None, threshold_dict
    
    elif extraction_params['type'] == 'transfer':
        
        rois = run_roi_transfer(extraction_params['transfer_data_path'],
                                extraction_params['transfer_type'],
                                experiment_info=extraction_params['experiment_conditions'],
                                imaging_info=extraction_params['imaging_information'])
        
        return cat_masks, cat_names, None, None, rois, None
    
    else:
       raise TypeError('ROI selection type not understood.') 

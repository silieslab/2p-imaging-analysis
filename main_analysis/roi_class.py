# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 12:54:12 2021

@author: smolina and Burak Gur
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import warn
import copy

class ROI_bg: 
    """A region of interest from an image sequence """
    
    def __init__(self,Mask = None, experiment_info = None,imaging_info = None): 
        """ 
        Initialized with a mask and optionally with experiment and imaging
        information
        """
        if (Mask is None):
            raise TypeError('ROI_bg: ROI must be initialized with a mask (numpy array)')
        if (experiment_info is not None):
            self.experiment_info = experiment_info
        if (imaging_info is not None):
            self.imaging_info = imaging_info
            
        self.mask = Mask
        self.uniq_id = id(self) # Generate a unique ID everytime 
        
    def __str__(self):
        return '<ROI:{_id}>'.format(_id = self.uniq_id)
    
    def __repr__(self):
        return '<ROI:{_id}>'.format(_id = self.uniq_id)
    
    def setCategory(self,Category):
        self.category = Category
        
    def set_z_depth(self,depth):
        self.z_depth = depth
        
    def setSourceImage(self, Source_image):
        
        if np.shape(Source_image) == np.shape(self.mask):
            self.source_image = Source_image
        else:
            raise TypeError('ROI_bg: source image dimensions has to match with\
                            ROI mask.')

    def set_extraction_type(self,extraction_type):
        self.extraction_type = extraction_type
    
    def showRoiMask(self, cmap = 'Pastel2',source_image = None):
        
        if (source_image is None):
            source_image = self.source_image
        curr_mask = np.array(copy.deepcopy(self.mask),dtype=float)
        curr_mask[curr_mask==0] = np.nan
        sns.heatmap(source_image,alpha=0.8,cmap = 'gray',cbar=False)
        sns.heatmap(curr_mask, alpha=0.6,cmap = cmap,cbar=False)
        plt.axis('off')
        plt.title(self)
        
    def calculateDf(self,method='mean',moving_avg = False, bins = 3):
        try:
            self.raw_trace
        except NameError:
            raise NameError('ROI_bg: for deltaF calculations, a raw trace \
                            needs to be provided: a.raw_trace')
            
        if method=='mean':
            df_trace = (self.raw_trace-self.raw_trace.mean(axis=0))/(self.raw_trace.mean(axis=0))
            self.baseline_method = method
        
        if moving_avg:
            self.df_trace = movingaverage(df_trace, bins)
        else:
            self.df_trace = df_trace
            
        return self.df_trace
            
    def plotDF(self, line_w = 1, adder = 0,color=plt.cm.Dark2(0)):
        
        plt.plot(self.df_trace+adder, lw=line_w, alpha=.8,color=color)
       
        try:
            self.stim_info['output_data']
            stim_frames = self.stim_info['output_data'][:,7]  # Frame information
            stim_vals = self.stim_info['output_data'][:,3] # Stimulus value
            uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
            stim_vals = stim_vals[uniq_frame_id]
            # Make normalized values of stimulus values for plotting
            
            stim_vals = (stim_vals/np.max(np.unique(stim_vals))) \
                *np.max(self.df_trace+adder)
            plt.plot(stim_vals,'--', lw=1, alpha=.6,color='k')
        except KeyError:
            print('No raw stimulus information found')
        
    
        
    def appendTrace(self, trace, epoch_num, trace_type = 'whole'):
        #JC: trace_type: whole means taking the whole trace with baseline
        #response means trace without baseline
        
        if trace_type == 'whole':
            try:
                self.whole_trace_all_epochs
            except AttributeError:
                self.whole_trace_all_epochs = {}
                
            
            self.whole_trace_all_epochs[epoch_num] = trace
        
        elif trace_type == 'response':
            try:
                self.resp_trace_all_epochs
            except AttributeError:
                self.resp_trace_all_epochs = {}
                
            self.resp_trace_all_epochs[epoch_num] = trace
            
    def appendStimInfo(self, Stim_info ,raw_stim_info = None):
        
        self.stim_info = Stim_info
        self.stim_name = Stim_info['stim_name']
        
        if (raw_stim_info is not None):
            # This part is now stored in the stim_info already but keeping it 
            # for backward compatibility.
            self.raw_stim_info = raw_stim_info
           
    def findMaxResponse_all_epochs(self):
        try:
            self.resp_trace_all_epochs
        except AttributeError:
            raise AttributeError('ROI_bg: for finding maximum responses \
                            "resp_trace_all_epochs" has to be appended by \
                            appendTrace() method ')
            
        
        
        self.max_resp_all_epochs = \
            np.empty(shape=(int(self.stim_info['EPOCHS']),1)) #Seb: epochs_number --> EPOCHS
        
        self.max_resp_all_epochs[:] = np.nan
        
        for epoch_idx in self.resp_trace_all_epochs:
            self.max_resp_all_epochs[epoch_idx] = np.nanmax(self.resp_trace_all_epochs[epoch_idx])
        
        self.max_response = np.nanmax(self.max_resp_all_epochs)
        self.max_resp_idx = np.nanargmax(self.max_resp_all_epochs)
        
    def calculate_DSI_PD(self,method='PDND'):
        '''Calcuates DSI and PD of an ROI '''
        #What is DSI and PD? -JC

        try:
            self.max_resp_all_epochs
            self.max_resp_idx
            self.stim_info
            self.max_response
        except AttributeError:
            raise TypeError('ROI_bg: for finding DSI an ROI needs\
                                 max_resp_all_epochs and stim_info')
        def find_opp_epoch(self, current_dir, current_freq, current_epoch_type):
            required_epoch_array = \
                    (self.stim_info['epoch_dir'] == ((current_dir+180) % 360)) & \
                    (self.stim_info['epoch_frequency'] == current_freq) & \
                    (self.stim_info['stimtype'] == current_epoch_type)
            
            return np.where(required_epoch_array)[0]
        
        if method == 'PDND':
            # Finding the maximum response epoch properties
            current_dir = self.stim_info['epoch_dir'][self.max_resp_idx]
            current_freq = self.stim_info['epoch_frequency'][self.max_resp_idx]
            current_epoch_type = self.stim_info['stimtype'][self.max_resp_idx]
            
            if current_freq == 0:
                warn('ROI %s -- max response is not in a moving epoch...' % self.uniq_id)
                moving_epochs = np.where(self.stim_info['epoch_frequency']>0)[0]
                # Find the moving epoch with max response
                idx = np.nanargmax(self.max_resp_all_epochs[moving_epochs])
                max_epoch = moving_epochs[idx]
                max_resp = self.max_resp_all_epochs[max_epoch]
                
            else:
                
                max_epoch = self.max_resp_idx
                max_resp = self.max_response
            # Calculating the DSI
            
            opposite_dir_epoch = find_opp_epoch(self,current_dir, current_freq,
                                                current_epoch_type)
            DSI = (max_resp - self.max_resp_all_epochs[opposite_dir_epoch])/\
                (max_resp + self.max_resp_all_epochs[opposite_dir_epoch])
            self.DSI = DSI[0][0]
            
            self.PD = current_dir
            
        elif method =='vector':
            dirs = self.stim_info['epoch_dir'][self.stim_info['baseline_epoch']+1:] #start to count w/ 1 -JC
            resps = self.max_resp_all_epochs[self.stim_info['baseline_epoch']+1:]
            
            # Functions work with radians so convert
            xs= np.transpose(resps)*np.cos(np.radians(dirs))
            ys = np.transpose(resps)*np.sin(np.radians(dirs))
            x = (xs).sum()
            y = (ys).sum()
            DSI_vector = [x, y]
            cosine_angle = np.dot(DSI_vector, [1,0]) / (np.linalg.norm(DSI_vector) * np.linalg.norm([1,0]))
            
            # origin = [0], [0] # origin point
            # for idx,direction in enumerate(dirs):
            #     plt.quiver(origin[0],origin[1], xs[0][idx],ys[0][idx],
            #                color=plt.cm.Dark2(idx),
            #                label=str(direction),scale=2)
            # plt.quiver(origin[0],origin[1], x, y, color='r',scale=6)
            # plt.legend()
            
            angle = np.degrees(np.arccos(cosine_angle))
            if y<0:
                angle = 360 - angle
            self.DSI  = np.linalg.norm(DSI_vector)/np.max(resps)
            self.PD = angle
            
            
        
        
      
        
        
    def calculate_CSI(self, frameRate = None):
        
        
        try:
            self.resp_trace_all_epochs
            self.stim_info
            
        except AttributeError:
            raise TypeError('ROI_bg: for finding CSI an ROI needs\
                                 resp_trace_all_epochs and stim_info')
        # Find edge epochs
        edge_epochs = np.where(self.stim_info['stimtype']==50)[0]
        epochDur= self.stim_info['epochs_duration']
        
        self.edge_response = np.max(self.max_resp_all_epochs[edge_epochs])
        # Find the edge epoch with max response
        idx = np.nanargmax(self.max_resp_all_epochs[edge_epochs])
        max_edge_epoch = edge_epochs[idx]
        
        
        raw_trace = self.resp_trace_all_epochs[max_edge_epoch]
        trace = raw_trace
        # Filtering to decrease noise in max detection
#        b, a = signal.butter(3, 0.3, 'low')
#        trace = signal.filtfilt(b, a, raw_trace)
        
        half_dur_frames = int((round(self.imaging_info['frame_rate'] * epochDur[max_edge_epoch]))/2)
        OFF_resp = np.nanmax(trace[:half_dur_frames])
        ON_resp = np.nanmax(trace[half_dur_frames:])
        CSI = (ON_resp-OFF_resp)/(ON_resp+OFF_resp)     #what is CSI? ->Contrast selective index -JC
        
        self.CSI = np.abs(CSI)      #set CSI of class -JC
        if CSI >0:      #local CSI -JC
            self.CS = 'ON'
        else:
            self.CS = 'OFF'
                
            
        
    def calculateTFtuning_BF(self):
        
        grating_epochs = np.where(((self.stim_info['stimtype'] == 61) | \
                                   (self.stim_info['stimtype'] == 46)) &\
                                   (self.stim_info['epoch_frequency'] > 0))[0]
        
        # If there are no grating epochs
        if grating_epochs.size==0:
            raise ValueError('ROI_bg: No grating epoch (stim type: 61 or 46 \
                                                        exists.')
            
        max_grating_epoch=np.nanargmax(self.max_resp_all_epochs[grating_epochs])      
        max_grating_epoch=grating_epochs[max_grating_epoch]            
        
        current_dir = self.stim_info['epoch_dir'][max_grating_epoch]
        current_epoch_type = self.stim_info['stimtype'][max_grating_epoch]
        
        # Finding all same direction moving grating epochs
        required_epoch_array = \
                (self.stim_info['epoch_dir'] == current_dir) & \
                (self.stim_info['stimtype'] == current_epoch_type)& \
                (self.stim_info['epoch_frequency'] > 0) 
        opposite_epoch_array = \
                (self.stim_info['epoch_dir'] == ((current_dir+180) % 360)) & \
                (self.stim_info['stimtype'] == current_epoch_type)& \
                (self.stim_info['epoch_frequency'] > 0) 
                
        self.TF_curve_stim = self.stim_info['epoch_frequency'][required_epoch_array]
        
        self.ND_TF_curve_stim = self.stim_info['epoch_frequency'][opposite_epoch_array]
        # Get it as integer indices
        req_epochs_PD = np.where(required_epoch_array)[0]
        self.TF_curve_resp = self.max_resp_all_epochs[req_epochs_PD]
        
        req_epochs_ND = np.where(opposite_epoch_array)[0]
        self.ND_TF_curve_resp = self.max_resp_all_epochs[req_epochs_ND]
        
        self.BF = self.stim_info['epoch_frequency'][max_grating_epoch]
        
#%% Other functions     
        
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def generate_ROI_instances(ROI_selection_dict, source_im,
                           experiment_info = None, imaging_info =None):
    """ Generates ROI_bg instances and adds the category information.

    Parameters
    ==========
    roi_masks : list
        A list of ROI masks in the form of numpy arrays.
        
    category_masks: list
        A list of category masks in the form of numpy arrays.
        
    category_names: list
        A list of category names.
        
    source_im : numpy array
        An array containing a representation of the source image where the 
        ROIs are found.
    
    Returns
    =======
    
    rois : list 
        A list containing instances of ROI_bg


    """
    #wrong documentation? -JC

    roi_masks = ROI_selection_dict['roi_masks']
    category_masks = ROI_selection_dict['cat_masks']
    category_names = ROI_selection_dict['cat_names']
    # Seb: coommented this    
    # if type(roi_masks) == sima.ROI.ROIList:
    #     roi_masks = list(map(lambda roi : np.array(roi)[0,:,:], roi_masks))
        
    # Generate instances of ROI_bg (class) from the masks
    rois = list(map(lambda mask : ROI_bg(mask, experiment_info = experiment_info,
                                    imaging_info=imaging_info), roi_masks))
                                    #lambda functions (local functions wo/ def) -JC

    def assign_region(roi, category_masks, category_names):
        """ Finds which layer the current mask is in"""
        for iLayer, category_mask in enumerate(category_masks):
            if np.sum(roi.mask*category_mask):          #0 == False -JC
                roi.setCategory(category_names[iLayer])
    
    # Add information            
    for roi in rois:
        assign_region(roi, category_masks, category_names)
        roi.setSourceImage(source_im)
        
    return rois

def separate_trials_ROI_v4(time_series,rois,stimulus_information,frameRate,
                           df_method = None, df_use = True,
                           max_resp_trial_len = 'max'):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces. Adds responses and whole traces into the ROI_bg
    instances.
    Parameters
    ==========
    time_series : numpy array
        Time series in the form of: frames x m x n (m & n are pixel dimensions)
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list.
        Each term in this list is a trial of the epoch. Trials consist of
        previous baseline epoch _ stimulus epoch _ following baseline epoch
        (if there is a baseline presentation)
        These terms have the following str: [[X, Y], [Z, D]] where
        first term is the trial beginning (first of first) and end
        (second of first), and second term is the baseline start
        (first of second) and end (second of second) for that trial.
    rois : list
        A list of ROI_bg instances.
    stimulus_information : list
        Stimulus related information is stored here.
    frameRate : float
        Image acquisiton rate.
    df_method : str
        Method for calculating dF/F defined in the ROI_bg class.
    plotting: bool
        If the user wants to visualize the masks and the traces for clusters.
    Returns
    =======
    wholeTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            baseline epoch - stimulus epoch - baseline epoch
    respTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
    baselineTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-
    """
    wholeTraces_allTrials_ROIs = {}
    respTraces_allTrials_ROIs = {}
    baselineTraces_allTrials_ROIs = {}

    # dF/F calculation
    for iROI, roi in enumerate(rois):
        roi.raw_trace = time_series[:,roi.mask].mean(axis=1)
        roi.calculateDf(method=df_method,moving_avg = True, bins = 3)
        if df_use:
            roi.base_dur = [] # Initialize baseline duration here for upcoming analysis

    trialCoor = stimulus_information['trial_coordinates']
    # Trial averaging by loooping through epochs and trials
    for iEpoch in trialCoor:
        currentEpoch = trialCoor[iEpoch]
        current_epoch_dur = stimulus_information['duration'][iEpoch]
        trial_numbers = len(currentEpoch)
        trial_lens = []
        resp_lens = []
        base_lens = []
        for curr_trial_coor in currentEpoch:
            current_trial_length = curr_trial_coor[0][1]-curr_trial_coor[0][0]
            trial_lens.append(current_trial_length)

            baselineStart = curr_trial_coor[1][0]
            baselineEnd = curr_trial_coor[1][1]
            base_len = baselineEnd - baselineStart

            base_lens.append(base_len)

            resp_start = curr_trial_coor[0][0]+base_len
            resp_end = curr_trial_coor[0][1]-base_len
            resp_lens.append(resp_end-resp_start)

        trial_len =  min(trial_lens)
        resp_len = min(resp_lens)
        base_len = min(base_lens)

        if not((max_resp_trial_len == 'max') or \
               (current_epoch_dur < max_resp_trial_len)):
            resp_len = int(round(frameRate * max_resp_trial_len))+1

        wholeTraces_allTrials_ROIs[iEpoch] = {}
        respTraces_allTrials_ROIs[iEpoch] = {}
        baselineTraces_allTrials_ROIs[iEpoch] = {}

        for iCluster, roi in enumerate(rois):

            # Baseline epoch is presented only when random value = 0 and 1
            if stimulus_information['random'] == 1:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(resp_len,
                                                         trial_numbers))
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(base_len,
                                                         trial_numbers))
            elif stimulus_information['random'] == 0:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                base_len  = np.shape(wholeTraces_allTrials_ROIs\
                                     [stimulus_information['baseline_epoch']]\
                                     [iCluster])[0]
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(int(frameRate*1.5),
                                   trial_numbers))
            else:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = None

            for trial_num , current_trial_coor in enumerate(currentEpoch):

                if stimulus_information['random'] == 1:
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]

                    baselineStart = current_trial_coor[1][0]
                    baselineEnd = current_trial_coor[1][1]

                    respStart = current_trial_coor[1][1]
                    epochEnd = current_trial_coor[0][1]

                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                        roi_resp = roi.df_trace[respStart:epochEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        roi_resp = roi.raw_trace[respStart:epochEnd]
                    try:
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    except ValueError:
                        new_trace = np.full((trial_len,),np.nan)
                        new_trace[:len(roi_whole_trace)] = roi_whole_trace.copy()
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace

                    respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_resp[:resp_len]
                    baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:base_len]
                elif stimulus_information['random'] == 0:

                    # If the sequence is non random  the trials are just separated without any baseline
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]

                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]


                    if iEpoch == stimulus_information['baseline_epoch']:
                        baseline_trace = roi_whole_trace[:base_len]
                        baseline_trace = baseline_trace[-int(frameRate*1.5):]
                        baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= baseline_trace
                    else:
                        baselineTraces_allTrials_ROIs[iEpoch][iCluster]\
                            [:,trial_num]= baselineTraces_allTrials_ROIs\
                            [stimulus_information['baseline_epoch']][iCluster]\
                            [:,trial_num]

                    try:
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    except ValueError:
                        new_trace = np.full((trial_len,),np.nan)
                        new_trace[:len(roi_whole_trace)] = roi_whole_trace.copy()
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace

                else:
                    # If the sequence is all random the trials are just separated without any baseline
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]

                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]

                    try:
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    except ValueError:
                        new_trace = np.full((trial_len,),np.nan)
                        new_trace[:len(roi_whole_trace)] = roi_whole_trace.copy()
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace

    for iEpoch in trialCoor:
        for iCluster, roi in enumerate(rois):

            # Appending trial averaged responses to roi instances only if
            # df is used
            if df_use:
                if stimulus_information['random'] == 0:
                    if iEpoch > 0 and iEpoch < len(trialCoor)-1:

                        wt = np.concatenate((np.nanmean(wholeTraces_allTrials_ROIs[iEpoch-1][iCluster],axis=1),
                                            np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1),
                                            np.nanmean(wholeTraces_allTrials_ROIs[iEpoch+1][iCluster],axis=1)),
                                            axis =0)
                        roi.base_dur.append(len(np.nanmean(wholeTraces_allTrials_ROIs[iEpoch-1][iCluster],axis=1)))
                    else:
                        wt = np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1)
                        roi.base_dur.append(0)
                elif stimulus_information['random'] == 1:
                    wt = np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1)
                    base_dur = frameRate * stimulus_information['baseline_duration']
                    roi.base_dur.append(int(round(base_dur)))
                else:
                    wt = np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1)


                roi.appendTrace(wt,iEpoch, trace_type = 'whole')
                roi.appendTrace(np.nanmean(respTraces_allTrials_ROIs[iEpoch][iCluster],axis=1),
                                  iEpoch, trace_type = 'response' )




    if df_use:
        print('Traces are stored in ROI objects.')
    else:
        print('No trace is stored in objects.')
    return (wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs,
            baselineTraces_allTrials_ROIs)


def separate_trials_ROI(time_series,rois,stimulus_information,frameRate,moving_avg, bins, 
                           df_method = None, df_first = True,
                           max_resp_trial_len = 'max'):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces. Adds responses and whole traces into the ROI_bg 
    instances.
    
    Parameters
    ==========
    time_series : numpy array
        Time series in the form of: frames x m x n (m & n are pixel dimensions)
    
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list.
        Each term in this list is a trial of the epoch. Trials consist of 
        previous baseline epoch _ stimulus epoch _ following baseline epoch
        (if there is a baseline presentation)
        These terms have the following str: [[X, Y], [Z, D]] where
        first term is the trial beginning (first of first) and end
        (second of first), and second term is the baseline start
        (first of second) and end (second of second) for that trial.
    
    rois : list
        A list of ROI_bg instances.
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    df_method : str
        Method for calculating dF/F defined in the ROI_bg class.
        
    plotting: bool
        If the user wants to visualize the masks and the traces for clusters.
        
    Returns
    =======
    wholeTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            baseline epoch - stimulus epoch - baseline epoch
            
    respTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-

    respTraces_allTrials_ROIs_raw : list containing np arrays
        Epoch list of raw time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-
            
    """
    #return dict not list? -JC

    wholeTraces_allTrials_ROIs = {}
    respTraces_allTrials_ROIs = {}
    respTraces_allTrials_ROIs_raw = {}
    baselineTraces_allTrials_ROIs = {}
    
    # dF/F calculation
    for iROI, roi in enumerate(rois):
        roi.raw_trace = time_series[:,roi.mask].mean(axis=1)
        roi.calculateDf(method=df_method,moving_avg = True, bins = 3)
        roi.base_dur = [] # Initialize baseline duration here for upcoming analysis
        
            
    trialCoor = stimulus_information['trial_coordinates']

    # Trial averaging by looping through epochs and trials
    for iEpoch in trialCoor:
        currentEpoch = trialCoor[iEpoch]
        current_epoch_dur = stimulus_information['duration'][iEpoch]
        trial_numbers = len(currentEpoch)
        trial_lens = []
        resp_lens = []
        base_lens = []
        for curr_trial_coor in currentEpoch:
            current_trial_length = curr_trial_coor[0][1]-curr_trial_coor[0][0]
            trial_lens.append(current_trial_length)
            
            baselineStart = curr_trial_coor[1][0]
            baselineEnd = curr_trial_coor[1][1]
            base_len = baselineEnd - baselineStart
            
            base_lens.append(base_len) 
            
            resp_start = curr_trial_coor[0][0]+base_len
            resp_end = curr_trial_coor[0][1]-base_len
            resp_lens.append(resp_end-resp_start)
        
        trial_len =  min(trial_lens)
        resp_len = min(resp_lens)
        base_len = min(base_lens)
        
        if not((max_resp_trial_len == 'max') or \
               (current_epoch_dur < max_resp_trial_len)):
            resp_len = int(round(frameRate * max_resp_trial_len))+1
            
        wholeTraces_allTrials_ROIs[iEpoch] = {}
        respTraces_allTrials_ROIs[iEpoch] = {}
        respTraces_allTrials_ROIs_raw[iEpoch] = {}
        baselineTraces_allTrials_ROIs[iEpoch] = {}
   
        for iCluster, roi in enumerate(rois):
            
            # Baseline epoch is presented only when random value = 0 and 1 
            if stimulus_information['random'] == 1:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(resp_len,
                                                         trial_numbers))
            
                respTraces_allTrials_ROIs_raw[iEpoch][iCluster] = np.zeros(shape=(resp_len,
                                                         trial_numbers))

                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(base_len,
                                                         trial_numbers))
            elif stimulus_information['random'] == 0:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs_raw[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))                               
                base_len  = np.shape(wholeTraces_allTrials_ROIs\
                                     [stimulus_information['baseline_epoch']]\
                                     [iCluster])[0]
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(int(frameRate*1.5),
                                   trial_numbers))
            else:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs_raw[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = None
            
            for trial_num , current_trial_coor in enumerate(currentEpoch):
                
                if stimulus_information['random'] == 1:
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    baselineStart = current_trial_coor[1][0]
                    baselineEnd = current_trial_coor[1][1]
                    
                    respStart = current_trial_coor[1][1]
                    epochEnd = current_trial_coor[0][1]
                    
                    if df_first:    #choose if df is averaged over all traces or if first averaged over trace and then df -JC
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                        roi_resp = roi.df_trace[respStart:epochEnd] #end baseline to end trial -JC
                        roi_resp_raw = roi.raw_trace[respStart:epochEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        roi_resp = roi.raw_trace[respStart:epochEnd]
                        roi_resp_raw = roi.raw_trace[respStart:epochEnd]
                    try:
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    except ValueError:
                        new_trace = np.full((trial_len,),np.nan)
                        new_trace[:len(roi_whole_trace)] = roi_whole_trace.copy()
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                            
                    respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_resp[:resp_len]
                    respTraces_allTrials_ROIs_raw[iEpoch][iCluster][:,trial_num]= roi_resp_raw[:resp_len]
                    baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:base_len]
                elif stimulus_information['random'] == 0:
                    
                    # If the sequence is non random  the trials are just separated without any baseline
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    if df_first:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                        roi_whole_trace_raw = roi.raw_trace[trialStart:trialEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        roi_whole_trace_raw = roi.raw_trace[trialStart:trialEnd]
                        
                    
                    if iEpoch == stimulus_information['baseline_epoch']:
                        baseline_trace = roi_whole_trace[:base_len]
                        baseline_trace = baseline_trace[-int(frameRate*1.5):]
                        baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= baseline_trace
                    else:
                        baselineTraces_allTrials_ROIs[iEpoch][iCluster]\
                            [:,trial_num] = baselineTraces_allTrials_ROIs\
                            [stimulus_information['baseline_epoch']][iCluster]\
                            [:,trial_num]
                    
                    try:
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                        respTraces_allTrials_ROIs_raw[iEpoch][iCluster][:,trial_num]= roi_whole_trace_raw[:trial_len]
                    except ValueError:
                        new_trace = np.full((trial_len,),np.nan)
                        new_trace[:len(roi_whole_trace)] = roi_whole_trace.copy()
                        new_trace_raw = np.full((trial_len,),np.nan)
                        new_trace_raw[:len(roi_whole_trace_raw)] = roi_whole_trace_raw.copy()
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                        respTraces_allTrials_ROIs_raw[iEpoch][iCluster][:,trial_num]= new_trace_raw
                        
                else:
                    # If the sequence is all random the trials are just separated without any baseline
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    if df_first:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                        roi_whole_trace_raw = roi.raw_trace[trialStart:trialEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        roi_whole_trace_raw = roi.raw_trace[trialStart:trialEnd]
                    
                    try:
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                        respTraces_allTrials_ROIs_raw[iEpoch][iCluster][:,trial_num]= roi_whole_trace_raw[:trial_len]
                    except ValueError:
                        new_trace = np.full((trial_len,),np.nan)
                        new_trace[:len(roi_whole_trace)] = roi_whole_trace.copy()
                        new_trace_raw = np.full((trial_len,),np.nan)
                        new_trace_raw[:len(roi_whole_trace_raw)] = roi_whole_trace_raw.copy()
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                        respTraces_allTrials_ROIs_raw[iEpoch][iCluster][:,trial_num]= new_trace_raw
                    
    for iEpoch in trialCoor:
        for iCluster, roi in enumerate(rois):
            
            # Appending trial averaged responses to roi instances only if 
            # df is used
            rt = np.nanmean(respTraces_allTrials_ROIs[iEpoch][iCluster],axis=1)

            if stimulus_information['random'] == 0:
                if iEpoch > 0 and iEpoch < len(trialCoor)-1:
                        
                    wt = np.concatenate((np.nanmean(wholeTraces_allTrials_ROIs[iEpoch-1][iCluster],axis=1),
                                            np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1),
                                            np.nanmean(wholeTraces_allTrials_ROIs[iEpoch+1][iCluster],axis=1)),
                                            axis =0) # Trial averaging
                    roi.base_dur.append(len(np.nanmean(wholeTraces_allTrials_ROIs[iEpoch-1][iCluster],axis=1)))
                else:
                    wt = np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1) # Trial averaging
                    roi.base_dur.append(0) 
                    #JC: average triles (e.g 5) for each epoch (e.g 0 and 1) for each
                    #ROI in each frame
                    #example: epoch:0, ROI:0, shape(array(55,5)), averaged: shape(array(55,)) -JC
            elif stimulus_information['random'] == 1:
                wt = np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1) # Trial averaging
                base_dur = frameRate * stimulus_information['baseline_duration']
                roi.base_dur.append(int(round(base_dur)))
            else:
                wt = np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1) # Trial averaging

            if not df_first: # Do df/f now, after trial averaging
                if df_method=='mean':
                    wt = (wt-np.mean(wt))/np.mean(wt) #JC: why not (wt-np.mean(wt))/np.mean(wt) = wt/np.mean(wt)-1?
                    rt = (rt-np.mean(rt))/np.mean(rt)   #JC: added -np.mean() to get df/f
                    roi.baseline_method = df_method
        
                if moving_avg:
                    wt = movingaverage(wt, bins)
                    rt = movingaverage(rt, bins)
                    
            roi.appendTrace(wt,iEpoch, trace_type = 'whole')
            roi.appendTrace(rt,iEpoch, trace_type = 'response' )
                    
    if df_first:
        print('df/f done BEFORE trial averaged')
    else:
        print('df/f done AFTER trial averaged.')
    return (wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs,respTraces_allTrials_ROIs_raw, 
            baselineTraces_allTrials_ROIs) 

def get_masks_image(rois):
    """ Generates an image of masks.

    Parameters
    ==========
    rois : list
        A list of ROI_bg instances.
        
    
    Returns
    =======
    
    roi_data_dict : np array
        A numpy array with masks depicted in different integers
    """   
    roi_masks_image = np.array(list(map(lambda idx_roi_pair : \
                             idx_roi_pair[1].mask.astype(float) * (idx_roi_pair[0]+1), 
                             list(enumerate(rois))))).sum(axis=0)
    
    roi_masks_image[roi_masks_image==0] = np.nan
    
    
    return roi_masks_image
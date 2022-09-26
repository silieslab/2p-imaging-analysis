#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:09:54 2019

@author: burakgur
"""
import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#import sima # Commented out by seb
import copy
import scipy
from warnings import warn
from scipy import signal
from roipoly import RoiPoly
from scipy.optimize import curve_fit
from scipy.stats import linregress
from skimage import filters
from scipy.stats.stats import pearsonr
from scipy import fft
from scipy.signal import blackman
import post_analysis_core as pac
import process_mov_core_reduced_msc_course as pmc
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
            dirs = self.stim_info['epoch_dir'][self.stim_info['baseline_epoch']+1:]
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
        CSI = (ON_resp-OFF_resp)/(ON_resp+OFF_resp)
        
        self.CSI = np.abs(CSI)
        if CSI >0:
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

#%% The Class ROI_bg ends here and some other functions relating with ROIs will follow

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def interpolate_signal(signal, sampling_rate, int_rate):
    """
    """

     #juan: corrected interpolation
    period=1/sampling_rate
    timeV=  np.linspace(period,(len(signal)+1)*period,num=len(signal))
    # Create an interpolated time vector in the desired interpolation rate
    timeVI=np.linspace(0.1,10,100) #logic (period already interpolated,duration of trace(S),period*duration(s)) #careful if you change int_rate. Hardcoded line for 10hz interpolation of a 10sec stimulus
    return np.interp(timeVI, timeV, signal)


def find_inverted(rois,stim_type = None):
    """
    Calculate pearson's correlation between responses and stimulus.
    
    """
    for roi in rois:
        if stim_type == '1Hz_gratings':
            fr = roi.imaging_info['frame_rate']
            baseline_frames_total = roi.stim_info['baseline_duration'] * fr
            baseline_frames_needed = int(baseline_frames_total-(roi.stim_info['baseline_duration']/2.0 * fr))
            
            baseline_m = roi.whole_trace_all_epochs[1][baseline_frames_needed:int(baseline_frames_total)].mean()
            
            response_m = roi.resp_trace_all_epochs[1].mean()
            
            diff = response_m - baseline_m
            if diff>0:
                roi.inverted = 0
            else:
                roi.inverted = 1    
            
        else:
            raise NameError('Stimulus type not found.')

    return rois


def generate_ROI_instances(roi_masks, category_masks, category_names, source_im,
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
    # Seb: coommented this    
    # if type(roi_masks) == sima.ROI.ROIList:
    #     roi_masks = list(map(lambda roi : np.array(roi)[0,:,:], roi_masks))
        
    # Generate instances of ROI_bg from the masks
    rois = list(map(lambda mask : ROI_bg(mask, experiment_info = experiment_info,
                                    imaging_info=imaging_info), roi_masks))

    def assign_region(roi, category_masks, category_names):
        """ Finds which layer the current mask is in"""
        for iLayer, category_mask in enumerate(category_masks):
            if np.sum(roi.mask*category_mask):
                roi.setCategory(category_names[iLayer])
    
    # Add information            
    for roi in rois:
        assign_region(roi, category_masks, category_names)
        roi.setSourceImage(source_im)
        
    return rois

#%%
def data_to_list(rois, data_name_list):
    """ Generates a dictionary with desired variables from ROIs.

    Parameters
    ==========
    rois : list
        A list of ROI_bg instances.
        
    data_name_list: list
        A list of strings with desired variable names. The variables should be 
        written as defined in the ROI_bg class. 
        
    Returns
    =======
    
    roi_data_dict : dictionary 
        A dictionary with keys as desired data variable names and values as
        list of data.
    """   
    class my_dictionary(dict):  
  
        # __init__ function  
        def __init__(self):  
            self = dict()  
              
        # Function to add key:value  
        def add(self, key, value):  
            self[key] = value  
    
    roi_data_dict = my_dictionary()
    
    # Generate an empty dictionary
    for key in data_name_list:
        roi_data_dict.add(key, [])
    
    # Loop through ROIs and get the desired data            
    for iROI, roi in enumerate(rois):
        for key, value in roi_data_dict.items(): 
            if key in roi.__dict__.keys():
                value.append(roi.__dict__[key])
            else:
                value.append(np.nan)
    return roi_data_dict


def threshold_ROIs(rois, threshold_dict):
    """ Thresholds given ROIs and returns the ones passing the threshold.

    Parameters
    ==========
    rois : list
        A list of ROI_bg instances.
        
    threshold_dict: dict
        A dictionary with desired ROI_bg property names that will be 
        thresholded as keys and the corresponding threshold values as values. 
    
    Returns
    =======
    
    thresholded_rois : list 
        A list containing instances of ROI_bg which pass the thresholding step.
    """
    # If there is no threshold
    if threshold_dict is None:
        print('No threshold used.')
        return rois
    vars_to_threshold = threshold_dict.keys()
    
    roi_data_dict = data_to_list(rois, vars_to_threshold)
    
    pass_bool = np.ones((1,len(rois)))
    
    for key, value in threshold_dict.items():
        
        if type(value) == tuple:
            if value[0] == 'b':
                pass_bool = \
                    pass_bool * (np.array(roi_data_dict[key]).flatten() > value[1])
                
            elif value[0] == 's':
                pass_bool = \
                    pass_bool * (np.array(roi_data_dict[key]).flatten() < value[1])
            else:
                raise TypeError("Tuple first value not understood: should be 'b' for bigger than or 's' for smaller than")
                
        else:
            pass_bool = pass_bool * (np.array(roi_data_dict[key]).flatten() > value)
    
    pass_indices = np.where(pass_bool)[1]
    
    thresholded_rois = []
    for idx in pass_indices:
        thresholded_rois.append(rois[idx])
    
    return thresholded_rois

#%% 
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

#%%
def generate_colorMasks_properties(rois, prop = 'BF'):
    """ Generates images of masks depending on DSI CSI Rel and BF

    TODO: Is it possible to generate something independent?
    Parameters
    ==========
    rois : list
        A list of ROI_bg instances.
        
    
    Returns
    =======
    
    roi_data_dict : np array
        A numpy array with masks depicted in different integers
    """  
    if prop == 'BF':
        BF_image = np.zeros(np.shape(rois[0].mask))
        
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            BF_image = BF_image + (curr_mask * roi.BF)
        BF_image[BF_image==0] = np.nan
        
        return BF_image
    elif prop == 'CS':
        CSI_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_CS = roi.CS
            if curr_CS == 'OFF':
                curr_CSI = roi.CSI * -1
            else:
                curr_CSI = roi.CSI
            curr_mask = roi.mask.astype(int)
            CSI_image = CSI_image + (curr_mask * curr_CSI)
        CSI_image[CSI_image==0] = np.nan
        return CSI_image
    elif prop =='DSI':
        DSI_image  = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_DSI = roi.DSI
                
            curr_mask = roi.mask.astype(int)
            DSI_image = DSI_image + (curr_mask * curr_DSI)
        DSI_image[DSI_image==0] = np.nan
        return DSI_image
    elif prop =='PD':
        PD_image  = np.full(np.shape(rois[0].mask),np.nan)
        alpha_image  = np.full(np.shape(rois[0].mask),np.nan)
        for roi in rois:
            PD_image[roi.mask] = roi.PD
            alpha_image[roi.mask] = roi.DSI
        
        return PD_image
    
    elif prop == 'reliability':
        Corr_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            Corr_image = Corr_image + (curr_mask * roi.reliability)
            
        Corr_image[Corr_image==0] = np.nan
        return Corr_image
    
    elif prop == 'SNR':
        snr_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            snr_image = snr_image + (curr_mask * roi.SNR)
            
        snr_image[snr_image==0] = np.nan
        return snr_image
    elif prop == 'corr_fff':
        Corr_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            Corr_image = Corr_image + (curr_mask * roi.corr_fff)
            
        Corr_image[Corr_image==0] = np.nan
        return Corr_image
    elif prop == 'max_response':
        max_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            max_image = max_image + (curr_mask * roi.max_response)
            
        max_image[max_image==0] = np.nan
        return max_image
    elif prop == 'slope':
        
        slope_im = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            slope_im = slope_im + (curr_mask * roi.slope)
            
        slope_im[slope_im==0] = np.nan
        return slope_im
    
    else:
        raise TypeError('Property %s not available for color mask generation' % prop)
        return 0

#%%
def analyze_gratings_general(rois):
    # IMPORTANT INFO
    # Seb: function name was changed from: analyze_luminance_gratings >>> analyze_gratings_general
    # The idea is to make one singl function handling all type of grating stimulation


    # Seb: if this variable does NOT exist in the stim file, made it 0 (= one single direction)
    if 'epoch_dir' in rois[0].stim_info:
        epoch_dirs = rois[0].stim_info['epoch_dir']
    else:
        epoch_dirs = np.ndarray.tolist(np.zeros(rois[0].stim_info['EPOCHS']))

    epoch_dirs_no_base= \
        np.delete(epoch_dirs,rois[0].stim_info['baseline_epoch'])
    epoch_types = rois[0].stim_info['stimtype']

    epoch_luminances= np.array(rois[0].stim_info['input_data']['lum'],float)
    epoch_velocity = np.array(rois[0].stim_info['input_data']['velocity'],float)
    epoch_sWavelength = np.array(rois[0].stim_info['input_data']['sWavelength'],float)
    epoch_TF= epoch_velocity/epoch_sWavelength

    if epoch_types[1] == 'noisygrating':
        epoch_SNR= np.array(rois[0].stim_info['input_data']['SNR'],float)
        

        
    for roi in rois:
        roi_dict = {}
        
        req_epochs = [e == rois[0].stim_info['stimtype'][-1] for e in epoch_types] # Seb: selecting epochs of interest based on the name of the last epoch in the stimulus input file
        if int(rois[0].stim_info['random']) == 1:
            req_epochs[0] = False # Seb: first epoch is for the baseline, not for analyzing any response
        #if rois[0].stim_info['stimtype'][0] != rois[0].stim_info['stimtype'][1]:
        #    req_epochs[0] = False
        #if rois[0].stim_info['stimtype'][0] == 'circle':
        #    req_epochs[0] = False #JC:4 times doing the same thing? commented it out
        
        roi_dict['luminance'] = epoch_luminances[req_epochs]
        roi_dict['deltaF'] = np.array(map(float,roi.max_resp_all_epochs[req_epochs]))
        roi_dict['TF'] = epoch_TF[req_epochs]
        # Specific variable based on the typ of stimulation for buiding a future heat map between this variable and TF
        if epoch_types[1] == 'lumgrating':
            variable_name = 'lum'

        elif epoch_types[1] == 'noisygrating':
            roi_dict['SNR'] = epoch_SNR[req_epochs]
            variable_name = 'SNR'

        elif epoch_types[1] == 'TFgrating':
            variable_name = 'TF'

        #Seb: fft analysis 
        #epochs_roi_data= roi.whole_trace_all_epochs # try also just with roi.resp_trace_all_epochs
        epochs_roi_data = roi.resp_trace_all_epochs
        
        amp_fft = []
        for idx, epoch in enumerate(epochs_roi_data):
            curr_trace = epochs_roi_data[epoch]
            N = len(curr_trace) # frames or total number of points (aka sample rate)
                
            # FFT and power spectra calculations
            period = 1.0 / roi.imaging_info['frame_rate']
            yf = fft.fft(curr_trace) #JC: added fft. because the module changed and now 
                                    #we need it to get the function
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = np.blackman(N)
            ywf = fft.fft((curr_trace-curr_trace.mean())*w)

            # "X" Hz sinusoidal as reference
            Lx = N/roi.imaging_info['frame_rate'] # Duration in seconds
            X_Hz = round(roi_dict['TF'][idx],2)
            f = X_Hz * np.rint(Lx) # X_hz 
            amp = 0.5 # desired amplitude
            x = np.arange(N)
            y = amp * np.sin(2 * np.pi * f * x / N)
            yf_ref = fft.fft(y) #fft values ref
            # yf_theo_ref = 2.0*np.abs(fft_values_ref/N)
            # mitigate spectral leakage
            w = np.blackman(N)
            ywf_ref = fft.fft((y-y.mean())*w)

            # Locating max peak of the reference frequency
            ref_trace_power = 2.0/N * np.abs(ywf_ref[1:N//2])
            m = max(ref_trace_power)
            max_idx = [i for i in range(len(ref_trace_power)) if ref_trace_power[i] == m][0]

            # Locating amplitude of the desired frequency in the response trace
            response_trace_power = 2.0/N * np.abs(ywf[1:N//2])
            temp_respose_amp = response_trace_power[max_idx]
            amp_fft.append(temp_respose_amp)

            # # plotting the spectrum
            # x_freqs = xf[1:N//2]

            # plt.close()
            # plt.plot(y)
            # plt.show()
            # plt.close()
            # plt.plot(x_freqs, response_trace_power, label='fft values')
            # plt.plot(x_freqs, ref_trace_power)
            # plt.plot(x_freqs[max_idx],temp_respose_amp, 'ro')
            # plt.title("Stimulated frequency (Hz): {}".format(X_Hz))
            # plt.show()
            # plt.close()

        # Saving the amplitude of the "X" hz component 
        roi.fft_X_hz_amp = amp_fft
        
        # Seb: if this variable does NOT exist in the stim file, create it from others
        if 'epoch_TF' in rois[0].stim_info:
            roi_dict['TF'] = roi.stim_info['epoch_TF'][req_epochs]
            
        else:
            temp_TF_list = []
            for i, value in enumerate(roi.stim_info['velocity']):
                #Seb: if statement to take care of non-grating epoch
                if value == 0.0 and roi.stim_info['sWavelength'][i] == 0.0:
                    temp_TF = 0.0
                    temp_TF_list.append(temp_TF)
                    continue
                temp_TF = value/roi.stim_info['sWavelength'][i]
                temp_TF_list.append(temp_TF)

            temp_TF_list = np.array(temp_TF_list)
            roi.stim_info['epoch_TF'] = temp_TF_list
            roi_dict['TF'] = temp_TF_list[req_epochs]


        # Creating a pandas dataframe for future heat map   
        df_roi = pd.DataFrame.from_dict(roi_dict)
        if epoch_types[1] == 'lumgrating':
            tfl_map = df_roi.pivot(index='TF',columns='luminance')
        elif epoch_types[1] == 'noisygrating':
            tfl_map = df_roi.pivot(index='TF',columns='SNR')
        elif epoch_types[1] == 'TFgrating':
            tfl_map = df_roi.pivot(index='TF',columns='luminance')
 
        roi.tfl_map= tfl_map
        roi.tfl_map_norm=(tfl_map-tfl_map.mean())/tfl_map.std()
        roi.BF = roi.stim_info['epoch_TF'][roi.max_resp_idx] 
        
        
        
        conc_trace = []
        for epoch in np.argwhere((roi.stim_info['epoch_TF'] == 1))[1:]:
            
            conc_trace=np.append(conc_trace,
                                 roi.whole_trace_all_epochs[float(epoch)],axis=0)
        roi.oneHz_conc_resp = conc_trace
        
    return rois



def analyze_gratings_1Hz(rois,int_rate = 10):  # Previous name: analyze_luminance_gratings_1Hz
    
    # Seb: if this variable does NOT exist in the stim file, made it 0 (= one single direction)
    if 'epoch_dir' in rois[0].stim_info:
        epoch_dirs = rois[0].stim_info['epoch_dir']
    else:
        epoch_dirs = np.ndarray.tolist(np.zeros(rois[0].stim_info['EPOCHS']))
    
    epoch_dirs_no_base= \
        np.delete(epoch_dirs,rois[0].stim_info['baseline_epoch'])
    epoch_types = rois[0].stim_info['stimtype']
    
    if epoch_types[-1] == 'lumgrating':
        epoch_luminances= np.array(rois[0].stim_info['input_data']['lum'],float)
    elif epoch_types[-1] == 'noisygrating':
        epoch_luminances= np.array(rois[0].stim_info['input_data']['SNR'],float)
        
    for roi in rois:

        # Seb: commented this out
        # if not('1D' in roi.stim_name):
        #     curr_pref_dir = \
        #         np.unique(epoch_dirs_no_base)[np.argmin(np.abs(np.unique(epoch_dirs_no_base)-roi.PD))]
        #     req_epochs = (epoch_dirs==curr_pref_dir) & (epoch_types != 11)
        # else:
        #     req_epochs = (epoch_types != 11)

        req_epochs = [e == rois[0].stim_info['stimtype'][-1] for e in epoch_types] # Seb: selecting epochs of interest based on the name of the last epoch in the stimulus input file
        if rois[0].stim_info['stimtype'][0] == rois[0].stim_info['stimtype'][1]:
            req_epochs[0] = False

        
        roi.luminances = epoch_luminances[req_epochs]            
        conc_trace = []
        roi.power_at_hz = np.zeros_like(roi.luminances)
        roi.base_power = np.zeros_like(roi.luminances)
        roi.baselines = np.zeros_like(roi.luminances)
        fr = roi.imaging_info['frame_rate']
        
        
        min_len = np.array(list(map(len, roi.resp_trace_all_epochs.values()))).min()
        traces = np.zeros((np.sum(req_epochs),min_len))
        
        ex_trace = roi.resp_trace_all_epochs[np.where(req_epochs)[0][0]][:min_len]
        int_len = len(interpolate_signal(ex_trace, 
                                     roi.imaging_info['frame_rate'],int_rate))
        int_traces =np.zeros((np.sum(req_epochs),int_len))
        
        
        min_len_wholeT = np.array(list(map(len, roi.whole_trace_all_epochs.values()))).min()
        traces_wholeT = np.zeros((np.sum(req_epochs),min_len_wholeT))
        
        ex_trace_wholeT = roi.whole_trace_all_epochs[np.where(req_epochs)[0][0]][:min_len_wholeT]
        int_len_wholeT = len(interpolate_signal(ex_trace_wholeT, 
                                     roi.imaging_info['frame_rate'],int_rate))
        int_traces_wholeT =np.zeros((np.sum(req_epochs),int_len_wholeT))
        mat_idx = 0
        for idx,epoch in enumerate(np.where(req_epochs)[0]):
            try:
                curr_freq = roi.stim_info['epoch_frequency'][epoch]
            except:
                curr_freq = roi.stim_info['epoch_TF'][epoch]
            curr_resp = roi.resp_trace_all_epochs[epoch]
            
            two_sec = int(2 * roi.imaging_info['frame_rate'])
            curr_whole = roi.whole_trace_all_epochs[epoch][two_sec:-1-two_sec]
            bg_resp = roi.whole_trace_all_epochs[epoch][two_sec:-two_sec+int(roi.imaging_info['frame_rate'])]
            bg_resp_mean = bg_resp.mean()
            roi.baselines[idx] = curr_resp[int(fr):].mean()-bg_resp_mean
            
            # Fourier analysis of baseline responses
            N = len(curr_whole)
            period = 1.0 / roi.imaging_info['frame_rate']
            x = np.linspace(0.0, N*period, N)
            yf = fft.fft(curr_whole)
            
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = blackman(N)
            ywf = fft.fft((curr_whole-curr_whole.mean())*w)
            # plt.plot(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]),
            #               label = '{l}'.format(l=roi.luminances[idx]))
            # plt.legend()
            base_p = 2.0/N * np.abs(ywf[1:N//2])
            req_idx = np.argmin(np.abs(xf-(1.0/6)))
            roi.base_power[idx] = base_p[req_idx]
            
            
            # Fourier analysis of sinusodial responses
            N = len(curr_resp)
            period = 1.0 / roi.imaging_info['frame_rate']
            x = np.linspace(0.0, N*period, N)
            yf = fft.fft(curr_resp)
            
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = blackman(N)
            ywf = fft.fft((curr_resp-curr_resp.mean())*w)
            # plt.plot(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]),
            #               label = '{l}'.format(l=roi.luminances[idx]))
            # plt.legend()
            power = 2.0/N * np.abs(ywf[1:N//2])
            req_idx = np.argmin(np.abs(xf-curr_freq))
            roi.power_at_hz[idx] = power[req_idx]
            
            # Concatenate trace
            conc_trace=np.append(conc_trace,
                                 roi.whole_trace_all_epochs[float(epoch)][two_sec:],axis=0)
            
            # Interpolation
            curr_trace = roi.resp_trace_all_epochs[epoch][:min_len]
            traces[mat_idx,:] = curr_trace
            int_traces[mat_idx,:] = interpolate_signal(curr_trace, 
                                 roi.imaging_info['frame_rate'],int_rate)
            
            curr_trace_wt = roi.whole_trace_all_epochs[epoch][:min_len_wholeT]
            traces_wholeT[mat_idx,:] = curr_trace_wt
            int_traces_wholeT[mat_idx,:] = interpolate_signal(curr_trace_wt, 
                                 roi.imaging_info['frame_rate'],int_rate)
            
            mat_idx +=1
            
       
        roi.int_rate = int_rate
        roi.grating_resp_traces = traces
        roi.grating_resp_traces_interpolated = int_traces
        
        roi.grating_whole_traces = traces_wholeT
        roi.grating_whole_traces_interpolated = int_traces_wholeT
             
        # plt.legend()
        # plt.title(roi.experiment_info['Genotype'])
        # plt.xlabel('Hz')
        # plt.ylabel('Signal')
        # plt.waitforbuttonpress()
        # plt.close('all')
        roi.conc_resp = conc_trace
        
        X = roi.luminances
        Y = roi.power_at_hz
        Z = roi.base_power
        
        roi.slope = linregress(X, np.transpose(Y))[0]
        roi.basePower_slope = linregress(X, np.transpose(Z))[0]
        roi.base_slope = linregress(X, np.transpose(roi.baselines))[0]
        
    return rois
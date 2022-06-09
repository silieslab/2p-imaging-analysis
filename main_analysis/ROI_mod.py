#!/usr/bin/env python2
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
import process_mov_core as pmc
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
            
        
    
        
def find_opp_epoch_roi(roi, current_dir, current_freq, current_epoch_type):
            required_epoch_array = \
                    (roi.stim_info['epoch_dir'] == ((current_dir+180) % 360)) & \
                    (roi.stim_info['epoch_frequency'] == current_freq) & \
                    (roi.stim_info['stimtype'] == current_epoch_type)  
            
            return np.where(required_epoch_array)[0]
def low_pass(trace, frame_rate, crit_freq=3,plot=False):
    """ Applies a 3rd order butterworth low pass filter for getting rid of noise
    """
    wn_norm = crit_freq / (frame_rate/2) #JC just commenting: desinging the critical frequency at which the signal will be filtered
    b, a = signal.butter(3, wn_norm, 'low') #JC just commenting: get filter
    filt_trace = signal.filtfilt(b, a, trace) #JC just commenting: applying the lowpass filter 
    
    if plot:
        fig1, ax1 = plt.subplots(2, 1, sharex=True, sharey=False)
        ax1[0].plot(trace,lw=0.4)
        ax1[1].plot(filt_trace,lw=0.4)

        
    
    return filt_trace
    
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
            
def fit_poly(trace, x = None, order=3):
    """ Fits a polynomial and return the fitted trace """
    if x is None:
        x = range(len(trace))
        
    z = np.polyfit(x, trace, order)
    p = np.poly1d(z)
    
    return p(x)
    


def calcualte_mask_1d_size(rois):
    """ Finds the coordinates of the masks and determins x and y size in pixels
    """
    
    for roi in rois:
        x = map(lambda coords : coords[0], np.argwhere(roi.mask))
        y = map(lambda coords : coords[1], np.argwhere(roi.mask))
        roi.x_size = np.max(x) - np.min(x) + 1
        roi.y_size = np.max(y) - np.min(y) + 1

def find_opp_epoch(roi, current_dir, current_freq, current_epoch_type):
            required_epoch_array = \
                    (roi.stim_info['epoch_dir'] == ((current_dir+180) % 360)) & \
                    (roi.stim_info['epoch_frequency'] == current_freq) & \
                    (roi.stim_info['stimtype'] == current_epoch_type)  
            
            return np.where(required_epoch_array)[0]
def ROI_find(rois, roi_prop, value):
    """ Finds in a list of rois, rois that has a certain value of a desired
        property and returns as another list.
    """
    
    
    return [roi for roi in rois if roi.__dict__[roi_prop] == value]

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
    rois = map(lambda mask : ROI_bg(mask, experiment_info = experiment_info,
                                    imaging_info=imaging_info), roi_masks)

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
    
    

    
def calculate_distance_from_region(rois):
    """
    """
    
    # Show the image
    fig = plt.figure()
    rois_image = get_masks_image(rois)
    
    plt.imshow(rois[0].source_image, interpolation='nearest', cmap='gray')
    plt.imshow(rois_image, alpha=0.5,cmap = 'tab20b')
    
    
    plt.title("Select a region")
    plt.show(block=False)
   
    
    # Draw ROI
    curr_roi = RoiPoly(color='r', fig=fig)
    plt.waitforbuttonpress()
    plt.pause(10)
    curr_mask = curr_roi.get_mask(rois_image)
    distance_mask=scipy.ndimage.morphology.distance_transform_edt(1-curr_mask)

    distances = list(map(lambda roi : np.min(distance_mask[roi.mask]), rois))
    
      
    for i, roi in enumerate(rois):
            roi.distance = distances[i]
    
    return distances,distance_mask


def calculate_edge_timing(rois):
    """
    """

    for roi in rois:
        # Find edge epochs
        edge_epochs = np.where(roi.stim_info['stimtype'] == 50)[0]
        # Find the edge epoch with max response
        idx = np.nanargmax(roi.max_resp_all_epochs[edge_epochs])
        max_edge_epoch = edge_epochs[idx]
        raw_trace = roi.resp_trace_all_epochs[max_edge_epoch]

        frameRate_approx = len(raw_trace)/8.0
        # Considering edge epochs are presented for 8s (change this later on for robustness)
        roi.edge_peak_t = np.argmax(raw_trace)/frameRate_approx

    edge_timings = list(map(lambda roi: roi.edge_peak_t, rois))


    return edge_timings
    

def make_ROI_tuning_summary(rois_df, roi,cmap='coolwarm', plot_x='reliability',plot_y='CSI'):
    
    from post_analysis_core import run_matplotlib_params
    
    curr_roi_props = rois_df[rois_df['uniq_id']==roi.uniq_id]
    plt.close('all')
    # Constructing the plot backbone, selecting colors
    colors = run_matplotlib_params()
    
    try:
        CS = roi.CS
    except AttributeError :
        CS = ''
    if roi.CS =='OFF':
        color = colors[3]
    else:
        color = colors[2]
    # Constructing the plot backbone
    fig = plt.figure(figsize=(9, 12))
    grid = plt.GridSpec(7, 8, wspace=2, hspace=1)
    
    # Mask
    plt.subplot(grid[:3,:3])
    roi.showRoiMask(cmap='PiYG')
    plt.title('%s PD: %d CS: %s' % (roi.uniq_id,roi.PD,CS))
    
    # plt.subplot(grid[:3,:3])
    # screen = np.zeros(np.shape(roi.RF_map))
    # screen[np.isnan(roi.RF_map)] = -0.1
    # curr_RF = np.full(np.shape(roi.RF_map), np.nan)
    # curr_RF[roi.RF_map_norm>0.99] = 1
    # # curr_RF[roi.RF_map_norm>0.99] = roi.__dict__['BF']
    # plt.imshow(roi.RF_map,alpha=1,cmap=cmap)   
    # plt.imshow(curr_RF,alpha=1,cmap='Greens_r',vmin=0,vmax=2,label='RF center')        
    
    # ax = plt.gca()
    # ax.set_title('ROI %s \nPD: %d CS: %s' % (roi.uniq_id,roi.PD,CS))
    # ax.axis('off')
    # ax.set_xlim(((np.shape(roi.RF_map)[0]-60)/2,(np.shape(roi.RF_map)[0]-60)/2+60))
    # ax.set_ylim(((np.shape(roi.RF_map)[0]-60)/2+60,(np.shape(roi.RF_map)[0]-60)/2))
    
    # Raw Trace
    plt.subplot(grid[0,3:])
    roi.plotDF(line_w=0.5,color = color)
    plt.title('Raw trace')
    
    # First property
    plt.subplot(grid[1:3,3:5])
    sns.scatterplot(x=plot_x, y=plot_y,alpha=.8,color='grey',
                    data =rois_df,legend=False,size=10)
    sns.scatterplot(x=plot_x, y=plot_y,color=color,
                    data =curr_roi_props,legend=False,size=7)
    
    plt.xlim(0, rois_df[plot_x].max()+0.3)
    plt.ylim(0, rois_df[plot_y].max()+0.3)
    
    
    # Tuning curve
    ax = plt.subplot(grid[1:3,5:])
    plt.plot(roi.TF_curve_stim,roi.TF_curve_resp,'-o',
             color = color,lw=3,markersize=10)
    
    ax.set(xscale="log")
    ax.set_title('TF tuning')
    ax.set_ylim(0,np.max(roi.TF_curve_resp)+0.2)
    ax.set_xlim((ax.get_xlim()[0],10)) 
    # Plot edge epochs
    grating_epochs = np.where(((roi.stim_info['stimtype'] == 61) | \
                             (roi.stim_info['stimtype'] == 46))\
                             & (roi.stim_info['epoch_frequency'] >0))[0]
    unique_freqs = np.unique(roi.stim_info['epoch_frequency'][grating_epochs])
    if len(unique_freqs) > 8:
        unique_freqs = unique_freqs[:8]
        
    adder =  4/len(unique_freqs)/2 + 1
    modn = len(unique_freqs)/2
    col_counter = np.repeat(range(2),len(unique_freqs)/2) *4
    
    
    for idx, freq in enumerate(unique_freqs):
        epochs = grating_epochs[roi.stim_info['epoch_frequency']\
                                [grating_epochs] == freq]
        ax = plt.subplot(grid[(3 + np.mod(idx,modn)):\
                              (3 + np.mod(idx,modn))+adder,col_counter[idx]\
                              :col_counter[idx]+4])
        
        max_epoch = epochs[np.argmax(roi.max_resp_all_epochs[epochs])]

        current_dir = roi.stim_info['epoch_dir'][max_epoch]
        current_freq = roi.stim_info['epoch_frequency'][max_epoch]
        current_epoch_type = roi.stim_info['stimtype'][max_epoch]
        opp_epoch = find_opp_epoch(roi, current_dir, current_freq, current_epoch_type)
        opp_epoch = opp_epoch[0]
        epochs_to_plot =[max_epoch, opp_epoch]
        
        for epoch in epochs_to_plot:
            if epoch==max_epoch:
                curr_color = color
            else:
                curr_color = colors[1]
                
            label_str = ('Dir: %d' % roi.stim_info['epoch_dir'][epoch])
            ax.plot(roi.whole_trace_all_epochs[epoch],label=label_str,lw=2,
                    color = curr_color)
            try :
                base_dur = roi.base_dur[epochs[0]]
                ax.plot([base_dur,base_dur],
                        [-0.5,np.ceil(roi.max_response)],'r')
                base_end = len(roi.resp_trace_all_epochs[epochs[0]]) +base_dur
                ax.plot([base_end,base_end],
                        [-0.5,np.ceil(roi.max_response)],'r')
     
            except AttributeError:
                print('No baseline duration found')
           
        ax.set_title('%.2f Hz'%freq)
        ax.set_ylim(-0.5,np.around(np.nanmax(roi.max_resp_all_epochs[grating_epochs]),1))
    fig.tight_layout()
    return fig
            
def calculate_distance_between_rois(rois):
    """
    Calculates the distance (in micrometers) between different ROIs.
    :param rois:
    :return distance_matrix:
    """
    distance_matrix = np.zeros(shape=(len(rois),len(rois)))
    for i, roi in enumerate(rois):
        distance_mask = scipy.ndimage.morphology.distance_transform_edt(1 - roi.mask)
        distance_mask = distance_mask*roi.imaging_info['pix_size']

        distance_matrix[i,:] = np.array(map(lambda roi: np.min(distance_mask[roi.mask]), rois))

    return distance_matrix

def fit_gauss(x,y):
    
    from scipy.optimize import curve_fit
    from scipy import asarray as exp
    
    
    n = len(x)                         
    mean = sum(x*y)/n                   
    sigma = sum(y*(x-mean)**2)/n        
    
    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
    
    popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])
    
    return gaus(x,*popt), popt, pcov

def map_RF(rois,edges=True,screen_dim = 60):
    """
    Maps the receptive field with a method based on the Fiorani et al 2014 paper:
    "Automatic mapping of visual cortex receptive fields: A fast and precise algorithm"
    
    """
    from scipy.ndimage.interpolation import rotate
    from scipy.stats import zscore
   
    screen_coords = np.linspace(0, screen_dim, num=screen_dim, endpoint=True) # degree of the screen

    for i, roi in enumerate(rois):
        lens = [len(v) for v in roi.resp_trace_all_epochs.values()]
        if edges and (roi.stim_name.find('LumDecLumInc') != -1):
            # cut the trace len to half for ON and OFF epochs
            trace_len = int(np.ceil(min(lens)/2.0)) 
        elif edges:
            print('Stimulus do not contain ON and OFF edges in single epoch.')
            
        pad = np.ceil((np.sqrt(2)*screen_dim-screen_dim)/2)
        dim = screen_dim+2*pad
        all_RFs = []

        
        for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
            if epoch_type == 50:
                curr_RF = np.full((int(dim), int(dim)),np.nan)
                b, a = signal.butter(3, 0.2, 'low')
                full_trace = signal.filtfilt(b, a,roi.resp_trace_all_epochs[epoch_idx])
                raw_trace = np.full((trace_len,),np.nan)
               
                if edges:
                    if roi.CS =='OFF':
                        raw_trace = full_trace[:trace_len]
                    else:

                        raw_trace[:trace_len] =\
                            full_trace[trace_len-np.mod(len(full_trace),2):
                                       trace_len-np.mod(len(full_trace),2)+trace_len]
                
                normalized = (raw_trace - min(raw_trace)) / \
                             (max(raw_trace) - min(raw_trace))
                # standardized = zscore(curr_response)
                resp_trace = raw_trace
                edge_speed = \
                    roi.stim_info['input_data']['Stimulus.stimtrans.mean'][epoch_idx]
                delay_frames = \
                    np.around(9.6/float(edge_speed) * roi.imaging_info['frame_rate'],0)
                resp_trace = np.roll(resp_trace,-int(delay_frames))
                roi_t_v = np.linspace(0, screen_dim, num=len(resp_trace), endpoint=True)
                i_resp = np.interp(screen_coords, roi_t_v, resp_trace)
                
                
                curr_direction = roi.stim_info['epoch_dir'][epoch_idx]

                back_projected = np.tile(i_resp, (len(i_resp),1) )
                back_projected[np.isnan(back_projected)] = 0
                # 90 degrees are rightwards w.r. to the fly so it shouldn't be turned
                # 0 degrees is upwards so 90-curr_dir
                # 
                rotated = rotate(back_projected+1, angle=np.mod(90-curr_direction,360))
                rotated[rotated==1000] = np.nan
                rotated[rotated==0] = np.nan
                rotated = rotated-1
                rot_dim = len(rotated)
                idx1 = int((dim-rot_dim)/2)
                idx2 = int((dim-rot_dim)/2+rot_dim)
                curr_RF[idx1 : idx2,idx1 : idx2] = rotated
                all_RFs.append(curr_RF)
        roi.RF_maps = all_RFs
        roi.RF_map = np.mean(roi.RF_maps, axis=0)
        roi.RF_map_norm = (roi.RF_map - np.nanmin(roi.RF_map)) / \
                             (np.nanmax(roi.RF_map) - np.nanmin(roi.RF_map))
        roi.RF_center_coords = np.argwhere(roi.RF_map==np.nanmax(roi.RF_map))[0]


    return rois
def map_RF_adjust_edge_time(rois,edges=True,delay_degrees=9.6,delay_use=False,
                            edge_props = {'45':71, '135':71,'225':71,'315':71,
                                          '0':51,'180':51,
                                          '90':75,'270':75}):
    """
    Maps the receptive field with a method based on the Fiorani et al 2014 paper:
    "Automatic mapping of visual cortex receptive fields: A fast and precise algorithm"
    
    
    Added the perspective correction by measuring the extend of the screen in
    all directions, following is the measurement from 31/01/2020:
        Horizontal (90 and 270 degrees direction) : 75 degrees
        Vertical (0 and 180) : 51 degrees
        Diagonals (45, 135, 225, 315) : 71 degrees
        
    Also added the delay that was measured as 9.6 degrees by the experiments
    that I have done. (delay between a 20dps edge and the center of RF probed
    with standing stripes)
        
    """
    from scipy.ndimage.interpolation import rotate
    from scipy.stats import zscore

    for i, roi in enumerate(rois):
        lens = [len(v) for v in roi.resp_trace_all_epochs.values()]
        if edges and (roi.stim_name.find('LumDecLumInc') != -1):
            # cut the trace len to half for ON and OFF epochs
            half_len = int(np.ceil(min(lens)/2.0)) 
        elif edges:
            print('Stimulus do not contain ON and OFF edges in single epoch.')
            
            
        # We need a common dimension for mapping all images and this should be 
        # the turned version of the maximum covered distance in the screen.
        # The turned version is maximum after a 45 degree turn (or for other
        # angles that have 45 deg in mod 90).
        dim = int(np.max(edge_props.values())*np.sqrt(2))
        all_RFs = []
        roi.delayed_RF_traces ={}
        roi.non_delayed_RF_traces ={}
        for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
            if epoch_type == 50:
                edge_speed = \
                    roi.stim_info['input_data']['Stimulus.stimtrans.mean'][epoch_idx] #Seb: what is store in this sitmulus input column?
                curr_direction = roi.stim_info['epoch_dir'][epoch_idx]
                try:
                    degrees_covered = edge_props[str(int(curr_direction))]
                    frames_needed = int(np.around((degrees_covered/float(edge_speed))\
                        * roi.imaging_info['frame_rate'],0))
                except KeyError:
                    raise KeyError('Edge direction not found: %s degs' % str(int(curr_direction)))
                
                
                curr_RF = np.full((int(dim), int(dim)),np.nan)
                b, a = signal.butter(3, 0.2, 'low')
                whole_t = roi.whole_trace_all_epochs[epoch_idx]
                whole_t = signal.filtfilt(b, a,whole_t)
                whole_t = whole_t -np.min(whole_t)+1
                resp_len = len(roi.resp_trace_all_epochs[epoch_idx])
                base_len = (len(whole_t)-resp_len)/2
                base_t = whole_t[:base_len]
                base_activity = np.mean(base_t)
                
                if delay_use:
                    delay_frames = \
                        np.around(delay_degrees/float(edge_speed) * roi.imaging_info['frame_rate'],0)
                    delayed_trace = np.roll(whole_t,-int(delay_frames))
                    roi.delay_used_in_RF = delay_degrees
                    full_trace =delayed_trace[base_len:base_len+resp_len]
                    
                    roi.delayed_RF_traces[epoch_idx] = full_trace
                    roi.non_delayed_RF_traces[epoch_idx] = whole_t[base_len:base_len+resp_len]
                    
                else:
                    roi.delay_used_in_RF = 0
                    full_trace =whole_t[base_len:base_len+resp_len]
                    
                
                
               
                if edges:
                    if roi.CS =='OFF':
                        raw_trace = full_trace[:frames_needed]
                    else:

                        raw_trace = full_trace[half_len:half_len+frames_needed]
                
                # Standardize responses so that DS responses dominate less
                sd = np.sqrt(np.sum(np.square(raw_trace))/(len(raw_trace)-1))
                normalized = (raw_trace - base_activity)/sd
                resp_trace = normalized
                
                
                # Need to map to the screen
                diagonal_dir = str(int(np.mod(curr_direction+90,360)))
                
                degree_needed = degrees_covered
                diag_dir_covered = edge_props[diagonal_dir]
                    
                    
                screen_coords = np.linspace(0, degree_needed, 
                                            num=degree_needed, endpoint=True)
                roi_t_v = np.linspace(0, degree_needed, 
                                      num=len(resp_trace), endpoint=True)
                i_resp = np.interp(screen_coords, roi_t_v, resp_trace)
                diagonal_dir = str(int(np.mod(curr_direction+90,360)))
                back_projected = np.tile(i_resp, (diag_dir_covered,1))
                back_projected[np.isnan(back_projected)] = 0
                # 90 degrees are rightwards w.r. to the fly so it shouldn't be turned
                # 0 degrees is upwards so 90-curr_dir
                # 
                rotated = rotate(back_projected+1, 
                                 angle=np.mod(90-curr_direction,360))
                rotated[rotated==0] = np.nan
                rotated = rotated-1
                idx1_1 = int((dim-rotated.shape[0])/2)
                idx1_2 = int((dim-rotated.shape[0])/2+rotated.shape[0])
                
                idx2_1 = int((dim-rotated.shape[1])/2)
                idx2_2 = int((dim-rotated.shape[1])/2+rotated.shape[1])
                
                
                curr_RF[idx1_1 : idx1_2,idx2_1 : idx2_2] = rotated
                all_RFs.append(curr_RF)
                
                
        roi.RF_maps = all_RFs
        roi.RF_map = np.mean(roi.RF_maps, axis=0)
        roi.RF_map_norm = (roi.RF_map - np.nanmin(roi.RF_map)) / \
                             (np.nanmax(roi.RF_map) - np.nanmin(roi.RF_map))
        roi.RF_center_coords = np.argwhere(roi.RF_map==np.nanmax(roi.RF_map))[0]
        
    return rois

def calculate_roi_delay(roi):
    
    delays_s = []
    delays_f = []
    for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
        if epoch_type == 50:
            
            curr_dir = roi.stim_info['epoch_dir'][epoch_idx]
            curr_freq = roi.stim_info['epoch_frequency'][epoch_idx]
            curr_type = roi.stim_info['stimtype'][epoch_idx]
            
            opp_epoch = find_opp_epoch_roi(roi,curr_dir,curr_freq,curr_type)
            
            b, a = signal.butter(3, 0.2, 'low')
            trace1 = signal.filtfilt(b, a,roi.whole_trace_all_epochs[epoch_idx])
            trace2 = signal.filtfilt(b, a,roi.whole_trace_all_epochs[int(opp_epoch)])
            trace2 = np.flip(trace2)
            
            delay_in_frames = np.abs(np.argmax(trace2) - np.argmax(trace1))
            delays_f.append(delay_in_frames)
            delay_in_s = delay_in_frames/roi.imaging_info['frame_rate']
            delays_s.append(delay_in_s)
        else:
            delays_f.append(None)
            delays_s.append(None)
    return delays_f, delays_s
            
    

                
                
def map_RF_adjust_stripe_time(rois,screen_props = {'45':74, '135':72,
                                                   '225':74,'315':72,
                                                   '0':53,'180':53,
                                                   '90':78,'270':78},
                              delay_use=False):
    """
    Maps the receptive field with a method based on the Fiorani et al 2014 paper:
    "Automatic mapping of visual cortex receptive fields: A fast and precise algorithm"
    
    
    Added the perspective correction by measuring the extend of the screen in
    all directions, following is the measurement from 02/2020:
        Horizontal (90 and 270 degrees direction) : 78 degrees
        Vertical (0 and 180) : 53 degrees
        Diagonals (45, 225) : 74 degrees
        Diagonals (135, 315) : 72 degrees
        
    # Delay correction not good as currently implemented here. Non DS neurons 
    # do not need delay correction
        
    """
    from scipy.ndimage.interpolation import rotate

    for i, roi in enumerate(rois):
        
        
                
        dim = int(np.max(screen_props.values())*np.sqrt(2))
        all_RFs = []
        
        (delays_f, delays_s) = calculate_roi_delay(roi)
        for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
            if epoch_type == 50:
                stripe_speed = \
                    roi.stim_info['input_data']['Stimulus.stimtrans.mean'][epoch_idx] #Seb: what is store in this sitmulus input column?
                curr_direction = roi.stim_info['epoch_dir'][epoch_idx]
                try:
                    degrees_covered = screen_props[str(int(curr_direction))]
                    frames_needed = int(np.around((degrees_covered/float(stripe_speed))\
                        * roi.imaging_info['frame_rate'],0))
                except KeyError:
                    raise KeyError('Edge direction not found: %s degs' % str(int(curr_direction)))
                
                
                curr_RF = np.full((int(dim), int(dim)),np.nan)
                b, a = signal.butter(3, 0.2, 'low')
                whole_t = signal.filtfilt(b, a,roi.whole_trace_all_epochs[epoch_idx])
                whole_t = whole_t -np.min(whole_t)
                resp_len = len(roi.resp_trace_all_epochs[epoch_idx])
                base_len = (len(whole_t)-resp_len)/2
                base_t = whole_t[:base_len]
                base_activity = np.mean(base_t)
                
                roi.delay_used = delay_use
                if delay_use:
                    delayed_trace = np.roll(whole_t,
                                            -int(delays_f[epoch_idx]))
                    raw_trace =delayed_trace[base_len:base_len+resp_len]
                else:
                    raw_trace =whole_t[base_len:base_len+resp_len]
                    
                # Standardize responses so that DS responses dominate less
                sd = np.sqrt(np.sum(np.square(raw_trace))/(len(raw_trace)-1))
                normalized = (raw_trace - base_activity)/sd
                resp_trace = normalized
            
                # Need to map to the screen
                diagonal_dir = str(int(np.mod(curr_direction+90,360)))
                
                degree_needed = degrees_covered
                diag_dir_covered = screen_props[diagonal_dir]
                    
                    
                screen_coords = np.linspace(0, degree_needed, 
                                            num=degree_needed, endpoint=True)
                roi_t_v = np.linspace(0, degree_needed, 
                                      num=len(resp_trace), endpoint=True)
                i_resp = np.interp(screen_coords, roi_t_v, resp_trace)
                diagonal_dir = str(int(np.mod(curr_direction+90,360)))
                back_projected = np.tile(i_resp, (diag_dir_covered,1))
                back_projected[np.isnan(back_projected)] = 0
                # 90 degrees are rightwards w.r. to the fly so it shouldn't be turned
                # 0 degrees is upwards so 90-curr_dir
                # 
                rotated = rotate(back_projected+1, 
                                 angle=np.mod(90-curr_direction,360))
                rotated[rotated==0] = np.nan
                rotated = rotated-1
                idx1_1 = int((dim-rotated.shape[0])/2)
                idx1_2 = int((dim-rotated.shape[0])/2+rotated.shape[0])
                
                idx2_1 = int((dim-rotated.shape[1])/2)
                idx2_2 = int((dim-rotated.shape[1])/2+rotated.shape[1])
                
                
                curr_RF[idx1_1 : idx1_2,idx2_1 : idx2_2] = rotated
                all_RFs.append(curr_RF)
                
                
        roi.RF_maps = all_RFs
        roi.RF_map = np.mean(roi.RF_maps, axis=0)
        roi.RF_map_norm = (roi.RF_map - np.nanmin(roi.RF_map)) / \
                             (np.nanmax(roi.RF_map) - np.nanmin(roi.RF_map))
        roi.RF_center_coords = np.argwhere(roi.RF_map==np.nanmax(roi.RF_map))[0]
        
    return rois


def map_RF_v2(rois,edges=True,screen_dim = 60,delay=9.6):
    """
    Maps the receptive field with a method based on the Fiorani et al 2014 paper:
    "Automatic mapping of visual cortex receptive fields: A fast and precise algorithm"
    
    """
    from scipy.ndimage.interpolation import rotate
    from scipy.stats import zscore
   
    screen_coords = np.linspace(0, screen_dim, num=screen_dim, endpoint=True) # degree of the screen

    for i, roi in enumerate(rois):
        lens = [len(v) for v in roi.resp_trace_all_epochs.values()]
        if edges and (roi.stim_name.find('LumDecLumInc') != -1):
            # cut the trace len to half for ON and OFF epochs
            trace_len = int(np.ceil(min(lens)/2.0)) 
        elif edges:
            print('Stimulus do not contain ON and OFF edges in single epoch.')
            
        pad = np.ceil((np.sqrt(2)*screen_dim-screen_dim)/2)
        dim = screen_dim+2*pad
        all_RFs = []
        all_RFs_no_delay = []

        
        for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
            if epoch_type == 50:
                curr_RF = np.full((int(dim), int(dim)),np.nan)
                curr_RF_nd=np.full((int(dim), int(dim)),np.nan)
                b, a = signal.butter(3, 0.2, 'low')
                whole_t = signal.filtfilt(b, a,roi.whole_trace_all_epochs[epoch_idx])
                whole_t = whole_t -np.min(whole_t)+1
                resp_len = len(roi.resp_trace_all_epochs[epoch_idx])
                base_len = (len(whole_t)-resp_len)/2
                base_t = whole_t[:base_len]
                base_activity = np.mean(base_t)
                
                
                full_trace =whole_t[base_len:base_len+resp_len]
                raw_trace = np.full((trace_len,),np.nan)
               
                if edges:
                    if roi.CS =='OFF':
                        raw_trace = full_trace[:trace_len]
                    else:

                        raw_trace[:trace_len] =\
                            full_trace[trace_len-np.mod(len(full_trace),2):
                                       trace_len-np.mod(len(full_trace),2)+trace_len]
                
                # Standardize responses so that DS responses dominate less
                sd = np.sqrt(np.sum(np.square(raw_trace))/(len(raw_trace)-1))
                normalized = (raw_trace - base_activity)/sd
                                
                # normalized = (raw_trace - min(raw_trace)) / \
                #              (max(raw_trace) - min(raw_trace))
                # standardized = zscore(curr_response)
                resp_trace = normalized
                
                # Fixing the delay of responses
                edge_speed = \
                    roi.stim_info['input_data']['Stimulus.stimtrans.mean'][epoch_idx] #Seb: what is store in this sitmulus input column?
                delay_frames = \
                    np.around(delay/float(edge_speed) * roi.imaging_info['frame_rate'],0)
                resp_trace = np.roll(resp_trace,-int(delay_frames))
                roi.resp_delay = delay
                resp_trace[-int(delay_frames):] = np.min(resp_trace)
                
                curr_direction = roi.stim_info['epoch_dir'][epoch_idx]
                # Need to map to the screen
                roi_t_v = np.linspace(0, screen_dim, 
                                      num=len(resp_trace), endpoint=True)
                i_resp = np.interp(screen_coords, roi_t_v, resp_trace)
                back_projected = np.tile(i_resp, (len(i_resp),1) )
                back_projected[np.isnan(back_projected)] = 0
                # 90 degrees are rightwards w.r. to the fly so it shouldn't be turned
                # 0 degrees is upwards so 90-curr_dir
                # 
                rotated = rotate(back_projected+1, 
                                 angle=np.mod(90-curr_direction,360))
                rotated[rotated==0] = np.nan
                rotated = rotated-1
                rot_dim = len(rotated)
                idx1 = int((dim-rot_dim)/2)
                idx2 = int((dim-rot_dim)/2+rot_dim)
                curr_RF[idx1 : idx2,idx1 : idx2] = rotated
                all_RFs.append(curr_RF)
                
                # Store the non-delayed original ones for a back up
                roi_t_v_nd = np.linspace(0, screen_dim, num=len(normalized), endpoint=True)
                i_resp_nd = np.interp(screen_coords, roi_t_v_nd, normalized)
                back_projected_nd = np.tile(i_resp_nd, (len(i_resp_nd),1) )
                back_projected_nd[np.isnan(back_projected_nd)] = 0
                # 90 degrees are rightwards w.r. to the fly so it shouldn't be turned
                # 0 degrees is upwards so 90-curr_dir
                # 
                rotated_nd = rotate(back_projected_nd+1, 
                                    angle=np.mod(90-curr_direction,360))
                rotated_nd[rotated_nd==0] = np.nan
                rotated_nd = rotated_nd-1
                rot_dim_nd = len(rotated_nd)
                idx1_nd = int((dim-rot_dim_nd)/2)
                idx2_nd = int((dim-rot_dim_nd)/2+rot_dim_nd)
                curr_RF_nd[idx1_nd : idx2_nd,idx1_nd : idx2_nd] = rotated_nd
                all_RFs_no_delay.append(curr_RF_nd)
                
        roi.RF_maps = all_RFs
        roi.RF_map = np.mean(roi.RF_maps, axis=0)
        roi.RF_map_norm = (roi.RF_map - np.nanmin(roi.RF_map)) / \
                             (np.nanmax(roi.RF_map) - np.nanmin(roi.RF_map))
        roi.RF_center_coords = np.argwhere(roi.RF_map==np.nanmax(roi.RF_map))[0]
        
        roi.RF_maps_no_delay = all_RFs_no_delay
        roi.RF_map_no_delay = np.mean(roi.RF_maps_no_delay, axis=0)
        roi.RF_center_coords_no_delay =\
            np.argwhere(roi.RF_map_no_delay==np.nanmax(roi.RF_map_no_delay))[0]


    return rois

def map_RF_no_delay(rois,edges=True,screen_dim = 60):
    """
    Maps the receptive field with a method based on the Fiorani et al 2014 paper:
    "Automatic mapping of visual cortex receptive fields: A fast and precise algorithm"
    
    """
    from scipy.ndimage.interpolation import rotate
   
    screen_coords = np.linspace(0, screen_dim, num=screen_dim, endpoint=True) # degree of the screen

    for i, roi in enumerate(rois):
        lens = [len(v) for v in roi.resp_trace_all_epochs.values()]
        if edges and (roi.stim_name.find('LumDecLumInc') != -1):
            # cut the trace len to half for ON and OFF epochs
            trace_len = int(np.ceil(min(lens)/2.0)) 
        elif edges:
            print('Stimulus do not contain ON and OFF edges in single epoch.')
            
        pad = np.ceil((np.sqrt(2)*screen_dim-screen_dim)/2)
        dim = screen_dim+2*pad
        all_RFs = []
        
        for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
            if epoch_type == 50:
                curr_RF = np.full((int(dim), int(dim)),np.nan)
                b, a = signal.butter(3, 0.2, 'low')
                whole_t = signal.filtfilt(b, a,roi.whole_trace_all_epochs[epoch_idx])
                whole_t = whole_t -np.min(whole_t)+1
                resp_len = len(roi.resp_trace_all_epochs[epoch_idx])
                base_len = (len(whole_t)-resp_len)/2
                base_t = whole_t[:base_len]
                base_activity = np.mean(base_t)
                
                full_trace =whole_t[base_len:base_len+resp_len]
                raw_trace = np.full((trace_len,),np.nan)
               
                if edges:
                    if roi.CS =='OFF':
                        raw_trace = full_trace[:trace_len]
                    else:

                        raw_trace[:trace_len] =\
                            full_trace[trace_len-np.mod(len(full_trace),2):
                                       trace_len-np.mod(len(full_trace),2)+trace_len]
                
                # Standardize responses so that DS responses dominate less
                sd = np.sqrt(np.sum(np.square(raw_trace))/(len(raw_trace)-1))
                normalized = (raw_trace - base_activity)/sd
                resp_trace = normalized
                
                
                curr_direction = roi.stim_info['epoch_dir'][epoch_idx]
                # Need to map to the screen
                roi_t_v = np.linspace(0, screen_dim, 
                                      num=len(resp_trace), endpoint=True)
                i_resp = np.interp(screen_coords, roi_t_v, resp_trace)
                back_projected = np.tile(i_resp, (len(i_resp),1) )
                back_projected[np.isnan(back_projected)] = 0
                # 90 degrees are rightwards w.r. to the fly so it shouldn't be turned
                # 0 degrees is upwards so 90-curr_dir
                # 
                rotated = rotate(back_projected+1, 
                                 angle=np.mod(90-curr_direction,360))
                rotated[rotated==0] = np.nan
                rotated = rotated-1
                rot_dim = len(rotated)
                idx1 = int((dim-rot_dim)/2)
                idx2 = int((dim-rot_dim)/2+rot_dim)
                curr_RF[idx1 : idx2,idx1 : idx2] = rotated
                all_RFs.append(curr_RF)
                
                
        roi.RF_maps = all_RFs
        roi.RF_map = np.mean(roi.RF_maps, axis=0)
        roi.RF_map_norm = (roi.RF_map - np.nanmin(roi.RF_map)) / \
                             (np.nanmax(roi.RF_map) - np.nanmin(roi.RF_map))
        roi.RF_center_coords = np.argwhere(roi.RF_map==np.nanmax(roi.RF_map))[0]
        
    return rois


# Gaussian 2d fit
def twoDgaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitTwoDgaussian(data):
    from scipy import optimize
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(twoDgaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p



def plot_RF(roi,cmap1='inferno',cmap2='viridis',center_plot = False, 
            center_val = 0.95):
    plt.close('all')
    colors = pac.run_matplotlib_params()
    fig1, ax1 = plt.subplots(ncols=3, nrows=3, figsize=(5, 4))
    ax = ax1.flatten()
    for idx , curr_axis in enumerate(ax):
        # Mean image

        if idx == 4:
  
            
            center_RF = copy.deepcopy(roi.RF_map_norm)
            center_RF[center_RF<center_val] =np.nan
            sns.heatmap(roi.RF_map_norm, cmap=cmap1, ax=ax[idx], cbar=False)
            if center_plot:
                sns.heatmap(center_RF,ax=ax[idx], cbar=False,alpha=.5,
                            cmap='Greens')
            
            
            
            # ax[idx].matshow(BT_map, cmap='viridis')
            #
            # BT_map = BT_map[]
            # BT_map[np.isnan(BT_map)] = np.nanmin(BT_map)
            # params = fitTwoDgaussian(BT_map)
            # fit = twoDgaussian(*params)
            # ax[idx].contour(fit(*np.indices(BT_map.shape)))

            
            ax[idx].axis('off')
            try:
                ax[idx].set_title('BT-RF - PD: {PD}'.format(PD = int(roi.PD)))
            except AttributeError:
                ax[idx].set_title('BT-RF - {cat}'.format(cat = roi.category))
            continue
        elif idx > 3:
            plt_idx = idx-1
        else:
            plt_idx = idx
        sns.heatmap(roi.RF_maps[plt_idx], ax=ax[idx], cmap=cmap2,cbar=False,
                    vmin=np.nanmin(roi.RF_maps),
                        vmax=np.nanmax(roi.RF_maps))
        
       
        
        epoch_dir = roi.stim_info['epoch_dir'][plt_idx+roi.stim_info['epoch_adjuster']]
        ax[idx].set_title('{deg}$^\circ$'.format(deg = str(int(epoch_dir))))
        ax[idx].axis('off')


    return fig1

def plot_RFs(rois, number=None, f_w =None,cmap='inferno',
             center_plot = False, center_val = 0.95):
    import random
    plt.close('all')
    colors = pac.run_matplotlib_params()
    if (number == None) or (f_w==None):
        f_w = 5
        if len(rois)>10:
            number=10
        else:
            number = len(rois)
    elif number > len(rois):
        number = len(rois)
    f_w = f_w*2
    # Randomize ROIs
    copy_rois = copy.deepcopy(rois)
    # random.shuffle(copy_rois)
        
        
    
    if number <= f_w/2:
        dim1= number
        dim2 = 1
    elif number/float(f_w/2) > 1.0:
        dim1 = f_w/2
        dim2 = int(np.ceil(number/float(f_w/2)))
    fig1, ax1 = plt.subplots(ncols=dim1, nrows=dim2, figsize=(dim1, dim2))
    ax = ax1.flatten()
    for idx, roi in enumerate(rois):
        if idx == number:
            break
        
        center_RF = copy.deepcopy(roi.RF_map_norm)
        center_RF[center_RF<center_val] =np.nan
        ax[idx].imshow(roi.RF_map_norm, cmap=cmap)
        if center_plot:
            ax[idx].imshow(center_RF,alpha=.5,
                        cmap='Greens')
        ax[idx].axis('off')
        ax[idx].set_xlim(((np.shape(roi.RF_map_norm)[1]-75)/2,(np.shape(roi.RF_map_norm)[0]-75)/2+75))
        ax[idx].set_ylim(((np.shape(roi.RF_map_norm)[0]-51)/2+51,(np.shape(roi.RF_map_norm)[0]-51)/2))
        try:
            ax[idx].set_title('PD: {pd}'.format(pd=int(roi.PD)),fontsize='xx-small')
        except AttributeError:
            a=0
    try:
        for ax_id in range(len(ax)-idx-1):
            ax[ax_id+idx].axis('off')
    except:
        a =1
    return fig1


def plot_RF_centers_on_screen(rois,prop = 'PD',cmap='hsv',
                              ylab='PD ($^\circ$)',lims=(0,360),rel_t=0):

    plt.close('all')
    colors = pac.run_matplotlib_params()

    screen = np.zeros(np.shape(rois[0].RF_map))
    rfs = np.full(np.shape(rois[0].RF_map), np.nan)
    
    screen[np.isnan(rois[0].RF_map)] = -0.1
    plt.imshow(screen, cmap='binary', alpha=.3)
    for idx, roi in enumerate(rois):
        if roi.reliability < rel_t:
            continue
        curr_RF = np.full(np.shape(roi.RF_map), np.nan)
        # curr_RF[roi.RF_center_coords[0]-1:roi.RF_center_coords[0]+1,
        #     roi.RF_center_coords[1]-1:roi.RF_center_coords[1]+1] = roi.__dict__[prop]
        # rfs[roi.RF_center_coords[0]-1:roi.RF_center_coords[0]+1,
        #     roi.RF_center_coords[1]-1:roi.RF_center_coords[1]+1] = roi.__dict__[prop]
        
        if prop == None:
            curr_RF[roi.RF_map_norm>0.95] = idx+1
        else:
            curr_RF[roi.RF_map_norm>0.95] = roi.__dict__[prop]
                
        
        plt.imshow(curr_RF,alpha=.5,cmap=cmap,vmin =lims[0],vmax=lims[1])
   
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(ylab)

    # sns.heatmap(screen,cmap='binary',alpha=.3,cbar=False)
    
    ax = plt.gca()
    ax.set_xlim(((np.shape(screen)[0]-60)/2,(np.shape(screen)[0]-60)/2+60))
    ax.set_ylim(((np.shape(screen)[0]-60)/2+60,(np.shape(screen)[0]-60)/2))
    # ax.set_xlabel('Screen $^\circ$')
    # ax.set_ylabel('$^\circ$')
    
    ax.axis('off')


    fig = ax.get_figure()
    return fig

def plot_RF_centers_on_screen_smooth(rois,prop = 'PD',cmap='hsv',
                              ylab='PD ($^\circ$)',lims=(0,360),rel_t=0):
    import scipy.ndimage as ndimage
    plt.close('all')
    colors = pac.run_matplotlib_params()

    screen = np.zeros(np.shape(rois[0].RF_map))
    rfs = np.full(np.shape(rois[0].RF_map), np.nan)
    
    screen[np.isnan(rois[0].RF_map)] = -0.1
    plt.imshow(screen, cmap='binary', alpha=.3)
    for roi in rois:
        if roi.reliability < rel_t:
            continue
        curr_RF = np.full(np.shape(roi.RF_map), np.nan)
        curr_RF_form = np.full((roi.RF_map.shape[0],roi.RF_map.shape[1]-1), np.nan)
        # curr_RF[roi.RF_center_coords[0]-1:roi.RF_center_coords[0]+1,
        #     roi.RF_center_coords[1]-1:roi.RF_center_coords[1]+1] = roi.__dict__[prop]
        # rfs[roi.RF_center_coords[0]-1:roi.RF_center_coords[0]+1,
        #     roi.RF_center_coords[1]-1:roi.RF_center_coords[1]+1] = roi.__dict__[prop]
        
        curr_RF[roi.RF_map_norm>0.95] = roi.__dict__[prop]
        curr_RF[np.isnan(curr_RF)] = 0
        
        curr_RF_form[ np.diff(curr_RF) != 0] = roi.__dict__[prop]
        
        img = ndimage.gaussian_filter(curr_RF, sigma=(5, 5), order=0)
        img[img==0] = np.nan
        plt.imshow(curr_RF_form,alpha=.5,cmap=cmap,vmin =lims[0],vmax=lims[1])
   
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(ylab)

    # sns.heatmap(screen,cmap='binary',alpha=.3,cbar=False)
    
    ax = plt.gca()
    ax.set_xlim(((np.shape(screen)[0]-60)/2,(np.shape(screen)[0]-60)/2+60))
    ax.set_ylim(((np.shape(screen)[0]-60)/2+60,(np.shape(screen)[0]-60)/2))
    # ax.set_xlabel('Screen $^\circ$')
    # ax.set_ylabel('$^\circ$')
    
    ax.axis('off')


    fig = ax.get_figure()
    return fig

def generate_time_delay_profile_2Dedges(rois):
    
    for roi in rois:
        epochDur= roi.stim_info['epochs_duration']
        max_epoch = roi.max_resp_idx
        roi.edge_start_loc = roi.stim_info['input_data']['Stimulus.stimtrans.amp'][max_epoch] #Seb: what is store in this sitmulus input column?
        roi.edge_speed = roi.stim_info['input_data']['Stimulus.stimtrans.mean'][max_epoch] #Seb: what is store in this sitmulus input column?
        half_dur_frames = int((round(roi.imaging_info['frame_rate'] * epochDur[max_epoch]))/2)
        trace = roi.resp_trace_all_epochs[roi.max_resp_idx]
        OFF_resp = trace[:half_dur_frames]
        ON_resp = trace[half_dur_frames:]
        if roi.CS =='ON':
            roi.two_d_edge_profile = ON_resp
        elif roi.CS =='OFF':
            roi.two_d_edge_profile = OFF_resp
    return rois


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    
def fit_1d_gauss(data_x, data_y):
    
    p0 = [np.max(data_y), np.argmax(data_y), 1]
    coeff, pcov = curve_fit(gauss, data_x, data_y, p0=p0)
    fit_trace = gauss(data_x, *coeff)
    
    residuals = data_y- fit_trace
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data_y-np.mean(data_y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return fit_trace, r_squared, coeff
def generate_RF_map_stripes(rois, screen_w = 60):
    
    screen_coords = np.linspace(0, screen_w, num=screen_w, endpoint=True) # degree of the screen
    for roi in rois:
        roi_t_v_stripe = np.linspace(0, 60, num=len(roi.max_resp_all_epochs[1:]), endpoint=True)
        
        roi.i_stripe_resp = np.interp(screen_coords, roi_t_v_stripe, 
                                  np.transpose(roi.max_resp_all_epochs[1:])[0])
        try:
            fit_trace, r_squared, coeff = fit_1d_gauss(screen_coords, roi.i_stripe_resp)
        except RuntimeError:
            print('Fit parameters not found... ROI fitting stopped {s}'.format(s=roi))
            roi.discard = True
            roi.stripe_gauss_profile = None
            roi.stripe_gauss_coeff = None
            roi.stripe_r_squared = None
            roi.stripe_gauss_fwhm = None
            continue
        roi.stripe_gauss_profile = fit_trace
        roi.stripe_gauss_coeff = coeff
        roi.stripe_gauss_fwhm = 2.355 * coeff[2]
        roi.stripe_r_squared = r_squared
        
    return rois

def generate_RF_profile_stripes(rois):
    
    for roi in rois:
        rf_profile = np.zeros(np.shape(roi.max_resp_all_epochs[1:]))
        for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
            if epoch_type == 57:
                
                whole_t = roi.whole_trace_all_epochs[epoch_idx]
                whole_t = whole_t -np.min(whole_t)+1
                resp_len = len(roi.resp_trace_all_epochs[epoch_idx])
                base_len = (len(whole_t)-resp_len)/2
                base_t = whole_t[:base_len]
                base_activity = np.mean(base_t)
                resp_trace =whole_t[base_len:base_len+resp_len]
                # Standardize responses so that DS responses dominate less
                sd = np.sqrt(np.sum(np.square(resp_trace))/(len(resp_trace)-1))
                normalized = (resp_trace - base_activity)/sd
                resp_trace = normalized
                rf_profile[epoch_idx-roi.stim_info['epoch_adjuster']]=\
                    np.max(resp_trace)
        roi.RF_profile = rf_profile
        roi.RF_profile_coords = \
            np.array(roi.stim_info['input_data']['Stimulus.stimtrans.amp'][1:],
                     float) #Seb: what is store in this sitmulus input column?
            
    a=np.array(map(lambda roi : roi.RF_profile, rois))
    all_rfs_sorted = list(a[:,:,0])
    all_rfs_sorted.sort(key=lambda trace : np.argmax(trace))
        
    lens=map(lambda roi : len(roi.resp_trace_all_epochs[roi.max_resp_idx]), 
          rois)
    max_traces =\
        map(lambda roi : roi.resp_trace_all_epochs[roi.max_resp_idx][:np.min(lens)], 
          rois)
    max_epoch_traces = np.vstack(max_traces)
    
    
    
    return rois, all_rfs_sorted,max_epoch_traces

def generate_time_delay_profile_combined(rois,screen_deg = 60):
    # TODO: Hard coded edge width bad practice!!!
    

    screen_coords = np.linspace(0, screen_deg, num=screen_deg, endpoint=True) # degree of the screen
    
    
    
    # Edge is presented in the full screen and not just 60 degrees of the visual field

    for roi in rois:
        
        roi_t_v_stripe = np.linspace(0, screen_deg, num=len(roi.max_resp_all_epochs[1:]), endpoint=True)
        
        
        diff = np.abs(int(roi.edge_start_loc)) - 40 -screen_deg/2
        start_frame = int(np.around((diff/float(roi.edge_speed)) * roi.imaging_info['frame_rate']))
        end_frame = start_frame + int(np.ceil((60/float(roi.edge_speed) * roi.imaging_info['frame_rate'])))
        roi_t_v_edge = \
            np.linspace(0, screen_deg, 
                        num=len(roi.two_d_edge_profile[start_frame:end_frame]),
                        endpoint=True)
        
        i_edge = np.interp(screen_coords, roi_t_v_edge, 
                           roi.two_d_edge_profile[start_frame:end_frame])
        roi.i_stripe_resp = np.interp(screen_coords, roi_t_v_stripe, 
                                  np.transpose(roi.max_resp_all_epochs[1:])[0])
        if roi.PD == 90: # Rotate the response if PD is 90 since  
            roi.i_edge_resp = i_edge[::-1]
        else:
            roi.i_edge_resp = i_edge
        try:
            fit_trace, r_squared, coeff = fit_1dtt_gauss(screen_coords, roi.i_edge_resp)
        except RuntimeError:
            print('Fit parameters not found... discarding {s}'.format(s=roi))
            roi.discard = True
            roi.edge_gauss_profile = None
            roi.edge_r_squared = None
            roi.edge_gauss_coeff = None
            roi.resp_delay_deg = None
            roi.resp_delay_fits_Rsq = None
            continue
        roi.edge_gauss_profile = fit_trace
        roi.edge_r_squared = r_squared
        roi.edge_gauss_coeff = coeff
        
        try:
            fit_trace, r_squared, coeff = fit_1d_gauss(screen_coords, roi.i_stripe_resp)
        except RuntimeError:
            print('Fit parameters not found... discarding {s}'.format(s=roi))
            roi.discard = True
            roi.stripe_gauss_profile = None
            roi.stripe_r_squared = None
            roi.stripe_gauss_coeff = None
            roi.resp_delay_deg = None
            roi.resp_delay_fits_Rsq = None
            continue
        roi.stripe_gauss_profile = fit_trace
        roi.stripe_r_squared = r_squared
        roi.stripe_gauss_coeff = coeff
        
        roi.resp_delay_deg = np.abs(np.argmax(roi.stripe_gauss_profile) - np.argmax(roi.edge_gauss_profile))
        roi.resp_delay_fits_Rsq = np.array([roi.edge_r_squared,roi.stripe_r_squared])
        roi.discard = False
        
    return rois
        
def filter_delay_profile_rois(rois,Rsq_t = 0.5):
    filtered_rois = []
    for roi in rois:
        if roi.discard:
            continue
        elif np.where(roi.resp_delay_fits_Rsq < Rsq_t)[0].size>0:
            continue
        else:
            filtered_rois.append(roi)
   
    return filtered_rois
def plot_delay_profile_examples(rois, number=None,f_w=None,lw=1.3,alpha=.7,
                                colors = None):
    import random
    plt.close('all')

    colorss = pac.run_matplotlib_params()
    plt.rcParams["axes.titlesize"] = 'x-small'
    
    rois_to_plot = []
    for idx, roi in enumerate(rois):
        if roi.discard:
           
            continue
        if np.where(roi.resp_delay_fits_Rsq < 0.5)[0].size>0:
            
            continue
        rois_to_plot.append(roi)
    # Randomize ROIs
    random.shuffle(rois)
        
    fig1, ax1 = plt.subplots(ncols=3, nrows=2,figsize=(10, 5))
    axs = ax1.flatten()
    
    for idx, ax in enumerate(axs):
        if idx == number:
            break
        try:
            aa = rois_to_plot[idx]
        except IndexError:
            break
        ax.plot(rois_to_plot[idx].edge_gauss_profile, '--k',lw=lw,alpha=alpha)
        ax.plot(rois_to_plot[idx].stripe_gauss_profile,'--k',lw=lw,alpha=alpha)
        ax.plot(rois_to_plot[idx].i_edge_resp, label='edge',lw=lw,alpha=alpha,
                color = colors[0])
        ax.plot(rois_to_plot[idx].i_stripe_resp, label='stripe',lw=lw,alpha=alpha,
                color = colors[1])
        limy_one= ax.get_ylim()[1]
        limy_zero = ax.get_ylim()[0]
        ax.plot([np.argmax(rois_to_plot[idx].edge_gauss_profile),
                  np.argmax(rois_to_plot[idx].edge_gauss_profile)],
                 [limy_zero,limy_one],'--',color = colors[0])
        
        ax.plot([np.argmax(rois_to_plot[idx].stripe_gauss_profile),
                  np.argmax(rois_to_plot[idx].stripe_gauss_profile)],
                 [limy_zero,limy_one],'--',color = colors[1])
        ax.set_ylim((limy_zero,limy_one))
        ax.set_xlabel('Screen ($^\circ$)')
        ax.set_ylabel('$\Delta F/F$')
       
        
        rsq1=round(rois_to_plot[idx].resp_delay_fits_Rsq[0],2)
        rsq2=round(rois_to_plot[idx].resp_delay_fits_Rsq[1],2)
        ax.set_title("Delay: {d}$^\circ$ Rsq {rsq1},{rsq2} PD {pd} CS {cs}".format(d=rois_to_plot[idx].resp_delay_deg,
                                                           rsq1=rsq1,rsq2=rsq2,
                                                           pd = rois_to_plot[idx].PD,
                                                           cs=roi.CS))
        ax.legend()
    fig1.tight_layout()
    return fig1
            
def transfer_masks(rois, properties,experiment_info = None, 
                   imaging_info =None,CS=None):
    """
    Generates new roi instances
    """
    new_rois = []
    for roi in rois:
        new_roi = ROI_bg(roi.mask, experiment_info = experiment_info,
                                    imaging_info=imaging_info)
        if CS != None:
            if not(roi.CS == CS):
                continue
                
        for prop in properties:
            # Note: Copy here is required otherwise it will just assign the pointer
            # which is dangerous if you want to use both rois in a script
            # that uses this function.
            try:
                new_roi.__dict__[prop] = copy.deepcopy(roi.__dict__[prop])
            except KeyError:
                print('Property:-{pr}- not found... Skipping property for this ROI\n'.format(pr=prop))
                continue
        new_rois.append(new_roi)
    print('ROI transfer successful.')
    return new_rois
        
def interpolate_signal(signal, sampling_rate, int_rate):
    """
    """

     #juan: corrected interpolation
    period=1/sampling_rate
    timeV=  np.linspace(period,(len(signal)+1)*period,num=len(signal))
    # Create an interpolated time vector in the desired interpolation rate
    timeVI=np.linspace(0.1,10,100) #logic (period already interpolated,duration of trace(S),period*duration(s)) #careful if you change int_rate. Hardcoded line for 10hz interpolation of a 10sec stimulus
    return np.interp(timeVI, timeV, signal)
    
def conc_traces(rois, interpolation = True, int_rate = 10):
    """
    Concatanates and interpolates traces.
    
    """
    for roi in rois:
        conc_trace = []
        stim_trace = []
        for idx, epoch in enumerate(range(roi.stim_info['EPOCHS'])): #Seb: epochs_number --> EPOCHS
            curr_stim = np.zeros((1,len(roi.whole_trace_all_epochs[epoch])))[0]
            curr_stim = curr_stim + idx
            stim_trace=np.append(stim_trace,curr_stim,axis=0)
            conc_trace=np.append(conc_trace,roi.whole_trace_all_epochs[epoch],axis=0)
        
        roi.conc_trace = conc_trace
        roi.stim_trace = stim_trace
        
        # Calculating correlation
        curr_coeff, pval = pearsonr(roi.conc_trace,roi.stim_trace)
        roi.corr_fff = curr_coeff
        roi.corr_pval = pval
        if interpolation:
            roi.int_con_trace = interpolate_signal(conc_trace, 
                                                   roi.imaging_info['frame_rate'], 
                                                   int_rate)
            roi.int_stim_trace = interpolate_signal(stim_trace, 
                                                    roi.imaging_info['frame_rate'], 
                                                   int_rate)
            roi.int_rate = int_rate
            
        
        
    return rois

def calculate_correlation(rois,stim_type = None):
    """
    Calculate pearson's correlation between responses and stimulus.
    
    """
    
    
    for roi in rois:
        if stim_type == '11LuminanceSteps':
            stim_to_correlate = roi.luminances
            resp = np.mean(roi.lum_resp_traces_interpolated,axis=1)
        elif stim_type == 'AB_steps':
            stim_to_correlate = roi.epoch_contrast_A_steps[1:]
            resp = roi.a_step_responses[1:]
            
            base_mean = roi.resp_traces_interpolated.mean(axis=0)[:50].mean()
            resp_mean = roi.resp_traces_interpolated.mean(axis=0)[70:100].mean()
            
            roi.mean_base_resp_diff = resp_mean - base_mean
        elif stim_type =='LuminanceEdges':
            fr = roi.imaging_info['frame_rate']
            stim_to_correlate = roi.luminances
            resp = []
            for trace in roi.resp_trace_all_epochs.values():
                resp.append(trace[int(-fr*4):].max())
            
        else:
            raise NameError('Stimulus type not found.')
            

        # Calculating correlation
        curr_coeff, pval = pearsonr(resp,stim_to_correlate)
        roi.correlation = curr_coeff
        roi.corr_pval = pval
        
    return rois

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



def transfer_props(roi, data_roi,transfer_props):
    """ 
    Transfers data from one roi to another one.
    """ 
    for prop in transfer_props:
        roi.__dict__[prop] = data_roi.__dict__[prop]
    return roi

def create_STF_maps(rois):
    
    """ Creates the spatiotemporal frequency maps """
    
    
    for roi in rois:
        roi_dict = {}
        pd_epochs = roi.stim_info['epoch_dir'][1:]==roi.PD
        roi_dict['SF'] = \
            np.array(roi.stim_info['input_data']['Stimulus.spacing'][1:]).astype(float)[pd_epochs]
        roi_dict['TF'] = roi.stim_info['epoch_frequency'][1:][pd_epochs]
            
        
        roi_dict['deltaF'] = np.array(map(float,roi.max_resp_all_epochs[1:]))[pd_epochs]
        
        df_roi = pd.DataFrame.from_dict(roi_dict)
        stf_map = df_roi.pivot(index='TF',columns='SF')
        roi.stf_map= stf_map
        roi.stf_map_norm=(stf_map-stf_map.mean())/stf_map.std()
        
        roi.BF = roi.stim_info['epoch_frequency'][roi.max_resp_idx]
    return rois
        
def plot_stf_map(roi,rois_df,plot_x='reliability',plot_y='CSI'):
    
    curr_roi_props = rois_df[rois_df['uniq_id']==roi.uniq_id]

    plt.close('all')
    from post_analysis_core import run_matplotlib_params
    # Constructing the plot backbone, selecting colors
    colors = run_matplotlib_params()
    try:
        CS = roi.CS
    except AttributeError :
        CS = ''
        color = colors[2]
    if roi.CS =='OFF':
        color = colors[3]
    else:
        color = colors[2]
    fig = plt.figure(figsize=(9, 12))
    
    # Mask
    plt.subplot(221)
    roi.showRoiMask(cmap='PiYG')
    plt.title('%s PD: %d CS: %s' % (roi.uniq_id,roi.PD,roi.CS))
    
    plt.subplot(222)
    sns.scatterplot(x=plot_x, y=plot_y,alpha=.8,color='grey',
                    data =rois_df,legend=False,size=20)
    sns.scatterplot(x=plot_x, y=plot_y,color=color,
                    data =curr_roi_props,legend=False,size=20)
    
    plt.xlim(0, rois_df[plot_x].max()+0.3)
    plt.ylim(0, rois_df[plot_y].max()+0.3)
    
    
    # Constructing the plot backbone
    plt.subplot(223)
    plt.title('STF map')
    ax=sns.heatmap(roi.stf_map, cmap='coolwarm',center=0,
                   xticklabels=np.array(roi.stf_map.columns.levels[1]).astype(int),
                   yticklabels=np.array(roi.stf_map.index),
                   cbar_kws={'label': '$\Delta F/F$'})
    ax.invert_yaxis()
    ax.invert_xaxis()
    plt.subplot(224)
    plt.title('STF map normalized')
    ax1=sns.heatmap(roi.stf_map_norm, cmap='coolwarm',center=0,
                   xticklabels=np.array(roi.stf_map_norm.columns.levels[1]).astype(int),
                   yticklabels=np.array(roi.stf_map_norm.index),
                   cbar_kws={'label': 'zscore'})
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    fig.tight_layout()
    
    return fig
def analyze_luminance_edges(rois,int_rate = 10):
    
    epoch_dirs = rois[0].stim_info['epoch_dir']
    epoch_dirs_no_base= \
        np.delete(epoch_dirs,rois[0].stim_info['baseline_epoch'])
    epoch_types = rois[0].stim_info['stimtype']
    epoch_file_l = np.array(rois[0].stim_info['input_data']['lum'],float)
    epoch_file_c = np.array(rois[0].stim_info['input_data']['contrast'],float)
    
    
    if 'OFF' in rois[0].stim_name:
        epoch_luminances= epoch_file_l * (1-epoch_file_c)
    elif 'ON' in rois[0].stim_name:
        epoch_luminances= epoch_file_l * (1+epoch_file_c)
        
    for roi in rois:
        
        # It needs to handle multiple types of same stimulus 
        if not('1D' in roi.stim_name):
            curr_pref_dir = \
                np.unique(epoch_dirs_no_base)[np.argmin(np.abs(np.unique(epoch_dirs_no_base)-roi.PD))]
            req_epochs = (epoch_dirs==curr_pref_dir) & (epoch_types != 11)
        else:
            req_epochs = (epoch_types != 11)
        
        
        roi.luminances = epoch_luminances[req_epochs]
        roi.edge_resps_abs = np.full((roi.luminances.shape), np.nan)
        roi.edge_resps = np.full((roi.luminances.shape), np.nan)
        roi.steady_state = np.full((roi.luminances.shape), np.nan)
        roi.baselines = np.full((roi.luminances.shape), np.nan)
        
        min_len = np.array(map(len, roi.resp_trace_all_epochs.values())).min()
        traces = np.zeros((np.sum(req_epochs),min_len))
        
        ex_trace = roi.resp_trace_all_epochs[np.where(req_epochs)[0][0]][:min_len]
        int_len = len(interpolate_signal(ex_trace, 
                                     roi.imaging_info['frame_rate'],int_rate))
        int_traces =np.zeros((np.sum(req_epochs),int_len))
        
        
        min_len_wholeT = np.array(map(len, roi.whole_trace_all_epochs.values())).min()
        traces_wholeT = np.zeros((np.sum(req_epochs),min_len_wholeT))
        
        ex_trace_wholeT = roi.whole_trace_all_epochs[np.where(req_epochs)[0][0]][:min_len_wholeT]
        int_len_wholeT = len(interpolate_signal(ex_trace_wholeT, 
                                     roi.imaging_info['frame_rate'],int_rate))
        int_traces_wholeT =np.zeros((np.sum(req_epochs),int_len_wholeT))
        
        
        mat_idx = 0
        for idx,  take in enumerate(req_epochs):
            if take:
                curr_trace = roi.resp_trace_all_epochs[idx][:min_len]
                traces[mat_idx,:] = curr_trace
                int_traces[mat_idx,:] = interpolate_signal(curr_trace, 
                                     roi.imaging_info['frame_rate'],int_rate)
                
                curr_trace_wt = roi.whole_trace_all_epochs[idx][:min_len_wholeT]
                traces_wholeT[mat_idx,:] = curr_trace_wt
                int_traces_wholeT[mat_idx,:] = interpolate_signal(curr_trace_wt, 
                                     roi.imaging_info['frame_rate'],int_rate)
                
                baseline = \
                    curr_trace[int(roi.imaging_info['frame_rate'])-int(roi.imaging_info['frame_rate']/3):int(roi.imaging_info['frame_rate'])].mean()
                roi.baselines[mat_idx] = baseline
                roi.edge_resps[mat_idx] = \
                    curr_trace[int(-roi.imaging_info['frame_rate']*4):].max() - baseline
                roi.edge_resps_abs[mat_idx] = \
                    curr_trace[int(-roi.imaging_info['frame_rate']*4):].max()
                roi.steady_state[mat_idx] = \
                    curr_trace_wt[int(-roi.imaging_info['frame_rate']*1):].mean()
                
                mat_idx +=1
        roi.int_rate = int_rate
        roi.edge_resp_traces = traces
        roi.edge_resp_traces_interpolated = int_traces
        
        roi.edge_whole_traces = traces_wholeT
        roi.edge_whole_traces_interpolated = int_traces_wholeT
        
        fp = roi.imaging_info['frame_rate']
        fp = np.tile(fp,len(roi.luminances))
        # roi_traces = np.array(map(lambda trace,fp : np.roll(trace[10:],len(trace[10:])/2 - \
        #                                     int(np.argmax(trace[10:]))),
        #              roi.edge_resp_traces_interpolated.copy(),fp))
            
        roi_traces = np.array(map(lambda trace,fp : np.roll(trace,len(trace)/2 - \
                                            int(np.argmax(trace[10:])+10)),
                     roi.edge_resp_traces_interpolated.copy(),fp))
    
            
        roi.max_aligned_traces = roi_traces.copy()
        roi_traces = np.concatenate(roi_traces)
        
        roi.concatenated_lum_traces = roi_traces.copy()
        X = roi.luminances
        Y = roi.edge_resps
        
        roi.slope = linregress(X, np.transpose(Y))[0]
        
    return rois

def analyze_A_B_step(rois,int_rate = 10):
    
    
    
    epoch_file_l = np.array(rois[0].stim_info['input_data']['bg'],float)
    #epoch_file_c = np.array(rois[0].stim_info['input_data']['contrast'],float)   # Seb: with new stimulus txt file, no need of contrast value
    #epoch_luminance_A_steps = epoch_file_l * (1+epoch_file_c)
    #epoch_luminance_B_steps = epoch_file_l * (1-epoch_file_c)

    epoch_lum_BG = epoch_file_l[0]
    epoch_luminance_A_steps = epoch_file_l
    epoch_luminance_B_steps = np.array(rois[0].stim_info['input_data']['fg'],float)
    epoch_contrast_A_steps = \
        (epoch_luminance_A_steps -epoch_lum_BG)/epoch_lum_BG
    epoch_contrast_B_steps = \
        (epoch_luminance_B_steps -epoch_luminance_A_steps)/epoch_luminance_A_steps
        
    epoch_contrast_B_steps_BGweber = \
        (epoch_luminance_B_steps -epoch_luminance_A_steps)/epoch_lum_BG
    
    for roi in rois:
        roi.epoch_lum_BG = epoch_lum_BG
        roi.epoch_luminance_A_steps = epoch_luminance_A_steps
        roi.epoch_luminance_B_steps = epoch_luminance_B_steps
        roi.epoch_contrast_A_steps = epoch_contrast_A_steps
        roi.epoch_contrast_B_steps = epoch_contrast_B_steps
        roi.epoch_contrast_B_steps_BGweber = epoch_contrast_B_steps_BGweber
                        
        min_len = np.array(map(len, roi.whole_trace_all_epochs.values())).min()
        traces = np.zeros((len(roi.whole_trace_all_epochs),min_len))
        ex_trace = roi.whole_trace_all_epochs[1][:min_len]
        int_len = len(interpolate_signal(ex_trace, 
                                     roi.imaging_info['frame_rate'],int_rate))
        int_traces =np.zeros((len(roi.whole_trace_all_epochs),int_len))
        mat_idx = 0
        
        roi.a_step_responses = np.full(np.shape(epoch_luminance_A_steps),np.nan)
        roi.b_step_responses = np.full(np.shape(epoch_luminance_B_steps),np.nan)
        
        roi.a_step_baseline_responses = np.full(np.shape(epoch_luminance_A_steps),np.nan)
        roi.b_step_baseline_responses = np.full(np.shape(epoch_luminance_B_steps),np.nan)
        
        roi.a_to_b_step_responses = np.full(np.shape(epoch_luminance_B_steps),np.nan)
        
        roi.a_step_responses = np.full(np.shape(epoch_luminance_A_steps),np.nan)
        roi.b_step_responses = np.full(np.shape(epoch_luminance_B_steps),np.nan)

        for epoch in roi.whole_trace_all_epochs:
            whole_t = roi.whole_trace_all_epochs[epoch][:min_len]
            resp_len = len(roi.resp_trace_all_epochs[epoch])
            base_len = (len(roi.whole_trace_all_epochs[epoch])-resp_len)/2
            base_t = \
                whole_t[base_len-(int(2*roi.imaging_info['frame_rate'])):base_len]
            base_activity = np.mean(base_t)
            resp_t = roi.resp_trace_all_epochs[epoch]
            A_t = resp_t[:int(len(resp_t)/2)]
            B_t = resp_t[int(len(resp_t)/2):]
            
            roi.a_step_responses[epoch] = np.max(A_t) - base_activity
            roi.b_step_responses[epoch] = np.max(B_t) - base_activity
            
            
            late_a = int(0.50 * roi.imaging_info['frame_rate'])
            early_b = int(1 * roi.imaging_info['frame_rate'])
            roi.a_step_baseline_responses[epoch] = \
                np.nanmean(A_t[-late_a:]) - base_activity
            roi.b_step_baseline_responses[epoch] = \
                np.nanmean(B_t[-late_a:]) - base_activity
                
            roi.a_to_b_step_responses[epoch] = np.nanmax(B_t[:early_b]) - np.nanmean(A_t[-late_a:])
            
            traces[mat_idx,:] = whole_t
            
            int_traces[mat_idx,:] = interpolate_signal(whole_t, 
                                     roi.imaging_info['frame_rate'],int_rate)
            mat_idx +=1
            
        roi.int_rate = int_rate
        roi.resp_traces = traces
        roi.resp_traces_interpolated = int_traces
        roi.AB_resp_max = traces.max()
        
        
    return rois

def sep_trial_compute_df(rois,df_method=None,df_base_dur=None,
                         max_resp_trial_len='max',filtering=False,
                         cf = 1):
    
    
    
    for roi in rois:
        fps = roi.imaging_info['frame_rate']
        if filtering:
            filtered = low_pass(roi.raw_trace.copy(), fps, 
                                        crit_freq=cf,plot=False)
            roi.raw_trace = filtered
        
        frameRate = roi.imaging_info['frame_rate']
        trialCoor = roi.stim_info['trial_coordinates']
        roi.whole_trace_all_epochs = {}
        roi.resp_trace_all_epochs = {}
        roi.base_dur = []
        # Trial averaging by loooping through epochs and trials
        for iEpoch in trialCoor:
            currentEpoch = trialCoor[iEpoch]
            current_epoch_dur = roi.stim_info['epoch_dur'][iEpoch]
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
                
       
            # Baseline epoch is presented only when random value = 0 and 1 
            if roi.stim_info['random'] == 1:
                wholeTraces_allTrials_ROIs = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs = np.zeros(shape=(resp_len,
                                                         trial_numbers))
                baselineTraces_allTrials_ROIs = np.zeros(shape=(base_len,
                                                         trial_numbers))
            elif roi.stim_info['random'] == 0:
                wholeTraces_allTrials_ROIs = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                base_len  = np.shape(wholeTraces_allTrials_ROIs\
                                     [roi.stim_info['baseline_epoch']])[0]
                baselineTraces_allTrials_ROIs = \
                    np.zeros(shape=(int(frameRate*1.5),trial_numbers))
            else:
                wholeTraces_allTrials_ROIs = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                baselineTraces_allTrials_ROIs = None
            
            for trial_num , current_trial_coor in enumerate(currentEpoch):
                
                if roi.stim_info['random'] == 1:
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    baselineStart = current_trial_coor[1][0]
                    baselineEnd = current_trial_coor[1][1]
                    
                    respStart = current_trial_coor[1][1]
                    epochEnd = current_trial_coor[0][1]
                    
                    
                    
                    if df_method == None:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        roi_resp = roi.raw_trace[respStart:epochEnd]
                        
                    elif df_method == 'Baseline_epoch':
                        base_f_len = int(frameRate * df_base_dur)
                        base_resp = roi.raw_trace[baselineEnd-base_f_len:baselineEnd].mean()
                        roi_whole_trace = (roi.raw_trace[trialStart:trialEnd] - base_resp)/base_resp
                        roi_resp = (roi.raw_trace[respStart:epochEnd] - base_resp)/base_resp
                    
                    try:
                        wholeTraces_allTrials_ROIs[:,trial_num]= roi_whole_trace[:trial_len]
                    except ValueError:
                        new_trace = np.full((trial_len,),np.nan)
                        new_trace[:len(roi_whole_trace)] = roi_whole_trace.copy()
                        wholeTraces_allTrials_ROIs[:,trial_num]= new_trace
                            
                    respTraces_allTrials_ROIs[:,trial_num]= roi_resp[:resp_len]
                    baselineTraces_allTrials_ROIs[:,trial_num]= roi_whole_trace[:base_len]
                elif roi.stim_info['random'] == 0:
                    a=1 
                    #TODO: FIX HERE
            wt = np.nanmean(wholeTraces_allTrials_ROIs,axis=1)
            roi.base_dur.append(df_base_dur)
            roi.appendTrace(wt,iEpoch, trace_type = 'whole')
            roi.appendTrace(np.nanmean(respTraces_allTrials_ROIs,axis=1),
                              iEpoch, trace_type = 'response' )
    return rois
        
                
                    
                    

def analyze_luminance_steps(rois,int_rate = 10):
    
    
    try:
        epoch_file_l = np.array(rois[0].stim_info['input_data']['lum'],float)
        epoch_file_c = np.array(rois[0].stim_info['input_data']['contrast'],float)
        epoch_luminances= epoch_file_l * (1-epoch_file_c)
    except:
        epoch_luminances= np.array(rois[0].stim_info['input_data']['bg'],float)
        
    for roi in rois:
        roi.luminances = epoch_luminances
        min_len = np.array(map(len, roi.resp_trace_all_epochs.values())).min()
        traces = np.zeros((len(roi.resp_trace_all_epochs),min_len))
        ex_trace = roi.resp_trace_all_epochs[0][:min_len]
        int_len = len(interpolate_signal(ex_trace, 
                                     roi.imaging_info['frame_rate'],int_rate))
        int_traces =np.zeros((len(roi.resp_trace_all_epochs),int_len))
        mat_idx = 0
        
        
        for epoch in roi.resp_trace_all_epochs:
            curr_trace = roi.resp_trace_all_epochs[epoch][:min_len]
            fps = roi.imaging_info['frame_rate']
            curr_trace = low_pass(curr_trace, fps, crit_freq=1,plot=False)
            traces[mat_idx,:] = curr_trace
            
            int_traces[mat_idx,:] = interpolate_signal(curr_trace, 
                                     roi.imaging_info['frame_rate'],int_rate)
            mat_idx +=1
            
        roi.int_rate = int_rate
        roi.lum_resp_traces = traces
        roi.lum_resp_traces_interpolated = int_traces
        roi.lum_resp_max = traces.max()
        
        
    return rois


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
        
        #Seb commented this out.
        # if not('1D' in roi.stim_name):
        #     curr_pref_dir = \
        #         np.unique(epoch_dirs_no_base)[np.argmin(np.abs(np.unique(epoch_dirs_no_base)-roi.PD))]
        #     req_epochs = (epoch_dirs==curr_pref_dir) & (epoch_types != 11)
        # else:
        #     req_epochs = (epoch_types != 11)
        req_epochs = [e == rois[0].stim_info['stimtype'][-1] for e in epoch_types] # Seb: selecting epochs of interest based on the name of the last epoch in the stimulus input file
        if int(rois[0].stim_info['random']) == 1:
            req_epochs[0] = False # Seb: first epoch is for the baseline, not for analyzing any response
        if rois[0].stim_info['stimtype'][0] != rois[0].stim_info['stimtype'][1]:
            req_epochs[0] = False
        if rois[0].stim_info['stimtype'][0] == 'circle':
            req_epochs[0] = False
        
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
            yf = fft(curr_trace)
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = np.blackman(N)
            ywf = fft((curr_trace-curr_trace.mean())*w)

            # "X" Hz sinusoidal as reference
            Lx = N/roi.imaging_info['frame_rate'] # Duration in seconds
            X_Hz = round(roi_dict['TF'][idx],2)
            f = X_Hz * np.rint(Lx) # X_hz 
            amp = 0.5 # desired amplitude
            x = np.arange(N)
            y = amp * np.sin(2 * np.pi * f * x / N)
            yf_ref = fft(y) #fft values ref
            # yf_theo_ref = 2.0*np.abs(fft_values_ref/N)
            # mitigate spectral leakage
            w = np.blackman(N)
            ywf_ref = fft((y-y.mean())*w)

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
        
        
        min_len = np.array(map(len, roi.resp_trace_all_epochs.values())).min()
        traces = np.zeros((np.sum(req_epochs),min_len))
        
        ex_trace = roi.resp_trace_all_epochs[np.where(req_epochs)[0][0]][:min_len]
        int_len = len(interpolate_signal(ex_trace, 
                                     roi.imaging_info['frame_rate'],int_rate))
        int_traces =np.zeros((np.sum(req_epochs),int_len))
        
        
        min_len_wholeT = np.array(map(len, roi.whole_trace_all_epochs.values())).min()
        traces_wholeT = np.zeros((np.sum(req_epochs),min_len_wholeT))
        
        ex_trace_wholeT = roi.whole_trace_all_epochs[np.where(req_epochs)[0][0]][:min_len_wholeT]
        int_len_wholeT = len(interpolate_signal(ex_trace_wholeT, 
                                     roi.imaging_info['frame_rate'],int_rate))
        int_traces_wholeT =np.zeros((np.sum(req_epochs),int_len_wholeT))
        mat_idx = 0
        for idx,epoch in enumerate(np.where(req_epochs)[0]):
            curr_freq = roi.stim_info['epoch_frequency'][epoch]
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
            yf = fft(curr_whole)
            
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = blackman(N)
            ywf = fft((curr_whole-curr_whole.mean())*w)
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
            yf = fft(curr_resp)
            
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = blackman(N)
            ywf = fft((curr_resp-curr_resp.mean())*w)
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

def analyze_luminance_freq_gratings(rois):
    '''CURRENTLY NOT WORKING '''
    
    
    epoch_dirs = rois[0].stim_info['epoch_dir']
    epoch_dirs_no_base= \
        np.delete(epoch_dirs,rois[0].stim_info['baseline_epoch'])
    epoch_types = rois[0].stim_info['stimtype']
    epoch_luminances= np.array(rois[0].stim_info['input_data']['lum'],float)
        
    for roi in rois:

        
        if not('1D' in roi.stim_name):
            curr_pref_dir = \
                np.unique(epoch_dirs_no_base)[np.argmin(np.abs(np.unique(epoch_dirs_no_base)-roi.PD))]
            req_epochs = (epoch_dirs==curr_pref_dir) & (epoch_types != 11)
        else:
            req_epochs = (epoch_types != 11)
            
        roi_dict = {}
        roi_dict['luminance'] = epoch_luminances[req_epochs]
        roi_dict['TF'] = roi.stim_info['epoch_frequency'][req_epochs]
        
        roi.luminances = epoch_luminances[req_epochs]            
        
        roi.power_at_hz = np.zeros_like(roi.luminances)
        for idx,epoch in enumerate(np.where(req_epochs)[0]):
            curr_freq = roi.stim_info['epoch_frequency'][epoch]
            if curr_freq > roi.imaging_info['frame_rate']//2:
                roi.power_at_hz[idx] = 0
                continue
                
            curr_resp = roi.resp_trace_all_epochs[epoch]
            
            # Fourier analysis
            N = len(curr_resp)
            period = 1.0 / roi.imaging_info['frame_rate']
            x = np.linspace(0.0, N*period, N)
            yf = fft(curr_resp)
            
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = blackman(N)
            ywf = fft(curr_resp*w)
            
            power = 2.0/N * np.abs(ywf[1:N//2])
            req_idx = np.argmin(np.abs(xf-curr_freq))
            roi.power_at_hz[idx] = power[req_idx]
            
            # Concatenate trace
        
        roi_dict['power'] = roi.power_at_hz[idx]
        df_roi = pd.DataFrame.from_dict(roi_dict)
        tfl_map = df_roi.pivot(index='TF',columns='luminance')
        roi.tfl_map= tfl_map
        
        roi.BF = roi.stim_info['epoch_frequency'][req_epochs][np.argmax(roi.power_at_hz)]
        
        
        conc_trace = []
        for epoch in np.argwhere((roi.stim_info['epoch_frequency'] == 1)):
            
            conc_trace=np.append(conc_trace,
                                 roi.whole_trace_all_epochs[float(epoch)],axis=0)
        roi.oneHz_conc_resp = conc_trace
        
        X = roi.luminances
        #Y = roi.power_at_hz[]
        
        roi.slope = linregress(X, np.transpose(Y))[0]
        
    return rois

def analyze_lum_con_gratings(rois):
    
    epoch_luminances = np.array(rois[0].stim_info['input_data']['lum'],float)
    epoch_contrasts = np.array(rois[0].stim_info['input_data']['contrast'],float)
    
    
    for roi in rois:
        roi_dict = {}
        
        roi_dict['luminance'] = epoch_luminances[1:]
        roi_dict['contrast'] = epoch_contrasts[1:]
            
        
        roi_dict['deltaF'] = np.array(map(float,roi.max_resp_all_epochs[1:]))
        
        df_roi = pd.DataFrame.from_dict(roi_dict)
        cl_map = df_roi.pivot(index='contrast',columns='luminance')
        roi.cl_map= cl_map
        roi.cl_map_norm=(cl_map-cl_map.mean())/cl_map.std()
                
        
    return rois


def reverse_correlation_analysis(rois,cf=2,filtering=True,
                                 poly_fitting=True,t_window=40,
                                 stim_up_rate=20, noise_type ='stripe' ):
    """ Reverse correlation analysis 
        
        JC: noise_type is a condition to define whether you used a 16x1 ('stripe') stimulus or 16x16 ('grit') one.
    """
    
    
    freq = stim_up_rate # The update rate of stimulus frames is 20Hz
    snippet = t_window

    for roi in rois:
        stim = roi.wn_stim
        # Stimulus related
        raw_stim_data = roi.stim_info['output_data']
        stim_vals_100Hz = raw_stim_data[:,6].astype(int) 
        image_frames_100Hz = raw_stim_data[:,7].astype(int)
        stim_time_100Hz = raw_stim_data[:,1].astype(float)
        epoch_vals = raw_stim_data[:,3].astype(int) # Epoch values
        frame_timings = roi.stim_info['frame_timings']
        # Response related stuff
        raw_signal = roi.raw_trace
        
        fps = roi.imaging_info['frame_rate']
        if filtering:
            filtered = low_pass(raw_signal, fps, crit_freq=cf,plot=False)
            trace = filtered.copy()
        else:
            trace = raw_signal.copy()
        
        
        
        
        if poly_fitting:
            fit_x = np.linspace(0,len(trace),len(trace))
            poly_vals = \
                np.polyfit(fit_x, trace, 4) #JC just commenting:fitting a polynaminal function
            fit_trace = np.polyval(poly_vals,fit_x) #JC just commenting: evalute the the fitted values with the poly_vals. But why?
            trace = trace-fit_trace
        trace = trace-trace.min()
        
        # df/f using the gray epochs #JC just commenting:2 sec gray epoch before WN stim
        # bg_frames = image_frames_100Hz[np.where(np.diff(epoch_vals==0))[0][0]]
        bg_frames = image_frames_100Hz[epoch_vals==0]
        mic_frames = bg_frames<len(frame_timings) #JC: in case stim_output is somehow longer than xml file
        bg_mean = trace[bg_frames[mic_frames]].mean()
        df_trace = (trace - bg_mean)/bg_mean
    
        newstimtimes, df, stimframes = pmc.interpolate_data_dyuzak(stim_time_100Hz,
                                                             stim_vals_100Hz,
                                                             df_trace,
                                                             frame_timings,
                                                             freq)
        
        # Finding where the stim frames is more preceeding frame length
        # Starting from these indices we will do the rev corr
        booleans = stimframes>=snippet
        # Padding stimulus frames by 0 (grey interleave) values
        padframes = np.concatenate(([0],stimframes,[0]))
        # Finding the points where frames shift between grey to frame numbers
        difs = np.diff(padframes>0)
        # Finding the indices of epoch start and end
        # rows = epoch no, columns = start and end
        epochind=np.where(difs==1)[0].reshape(-1,2)
        
        # Take the first epoch only since it contains a while WN stimulus
        analyzelimit = epochind[0,:]    #where to splice
        # Splicing data
        stimframesused = stimframes[analyzelimit[0]:analyzelimit[1]]
        dfused = df[analyzelimit[0]:analyzelimit[1]]
        
        # Used stimulus is sliced by using the frame information
        
        stimused = stim[np.min(stimframesused)-1:np.max(stimframesused),:,:]
        # Get unique frame numbers from trials
        uniquestim = np.unique(stimframesused)
        # Find frame nos grater than preceeding time window as a boolean matrix
        boolean2 = uniquestim >= snippet
        
        
        #centering df (mean substraction)
        
        centereddf = (dfused.T-np.mean(dfused)).T
        #centereddf = dfused[roi_ind] # no centering (mh, 11.02.19)
    
        #centering stimulus (mean substraction)
        stimused = stimused - np.mean(stimused)
        #stimused = stimused - 0.5 #shift around 0 (mh, 11.02.19) 

        # including noise_type condition #JC 16.03.2022
        if noise_type == 'stripe':
            stas = np.empty(shape=(snippet,stimused.shape[1]))
            avg = np.zeros(shape=(snippet,stimused.shape[1]))
        elif noise_type == 'grit':
            stas = np.empty(shape=(snippet,stimused.shape[1], stimused.shape[2]))
            avg = np.zeros(shape=(snippet,stimused.shape[1], stimused.shape[2]))

        # For loop iterates through different stimulus frame numbers
        # Finds the data where the specific frame is shown and calculates
        # sta
        for ii in range(snippet-1,len(uniquestim)):
            # Calculate means of responses to specific frame with ii index
            responsechunk = centereddf[np.where(stimframesused==uniquestim[ii])[0]]
            responsemean = np.mean(responsechunk) # JC: just one value why mean?

            # including noise_type condition #JC 16.03.2022
            if noise_type == 'stripe':
                # Create a tiled matrix for fast calculation
                response = np.tile(responsemean,(stimused.shape[1],snippet)).T
                # Find the stimulus values in the window and get tiled matrix
                stimsnip = stimused[uniquestim[ii]-snippet:uniquestim[ii],:,0] #JC: why not use the last column? (... ,:,:)?
            elif noise_type == 'grit':
                response = np.tile(responsemean,(stimused.shape[2],stimused.shape[1],snippet)).T
                stimsnip = stimused[uniquestim[ii]-snippet:uniquestim[ii],:,:]
            # Fast calculation is actually is a multiplication
            avg += np.multiply(response,stimsnip)
        sta = avg/np.sum(boolean2)     # Average with the number of additions
        roi.sta = sta
        
    return rois

def plot_STRFs(rois, f_w=None,number=None,cmap='coolwarm'):
    import random
    plt.close('all')
    #colors = pac.run_matplotlib_params() #JC: not used here, so I commented it out
    if (number == None) or (f_w==None):
        f_w = 5
        if len(rois)>10:
            number=10
        else:
            number = len(rois)
    elif number > len(rois):
        number = len(rois)
    f_w = f_w*2
    # Randomize ROIs
    copy_rois = copy.deepcopy(rois)
    random.shuffle(copy_rois)
    max_n=np.array(list(map(lambda roi : np.max(roi.sta), rois))).max()
    min_n=np.array(list(map(lambda roi : np.max(roi.sta), rois))).min()
        
    
    if number <= f_w/2:
        dim1= int(number) #JC added int here
        dim2 = 1
    elif number/float(f_w/2) > 1.0:
        dim1 = int(f_w/2) #JC added int here
        dim2 = int(np.ceil(number/float(f_w/2)))
    fig1, ax1 = plt.subplots(ncols=dim1, nrows=dim2, figsize=(dim1, dim2))
    ax = ax1.flatten()
    fig2, ax2 = plt.subplots(ncols=dim1, nrows=dim2, figsize=(dim1, dim2))
    ax2 = ax2.flatten()
    fig3, ax3 = plt.subplots(ncols=dim1, nrows=dim2, figsize=(dim1, dim2))
    ax3 = ax3.flatten()
    
    #JC: define which shape the stim had (and now the sta has)
    # e.g if sta.shape[1]=16 and sta.shape[2]=1 
    # then only one dimension was used (stripes)
    # or if sta.shape[1]=16 and sta.shape[2]=16, then checker (from Buraks code)
    sta_d1 = rois[0].sta.shape[1]
    sta_d2 = rois[0].sta.shape[2]

    for idx, roi in enumerate(rois):
        if idx == number:
            break
        
        if sta_d1 == 1:
            curr_sta = roi.sta[:,0,:].T
        elif sta_d2 ==1:
            curr_sta = roi.sta[:,:,0].T
        else:
            max_t = np.where(np.abs(roi.sta)==np.abs(roi.sta).max())[0][0] #JC for checker git time of max response
            curr_sta = roi.sta[max_t,:,:]
            plot_title = 'Space-Space plot'
            #git max coordinates for both space dimensions
            max_coords = np.array(np.where(np.abs(roi.sta)==np.abs(roi.sta).max())).astype(float)
            space_dim_1_max = int(max_coords[1])
            space_dim_2_max = int(max_coords[2])
            time_space1_sta = roi.sta[:,space_dim_1_max,:].T
            time_space2_sta = roi.sta[:,:,space_dim_2_max].T
            sns.heatmap(time_space1_sta, cmap=cmap, ax=ax2[idx], cbar=False,vmax=max_n,
                    center=0)
            ax2[idx].axis('off')
            sns.heatmap(time_space2_sta, cmap=cmap, ax=ax3[idx], cbar=False,vmax=max_n,
                        center=0)
            ax3[idx].axis('off')
        sns.heatmap(curr_sta, cmap=cmap, ax=ax[idx], cbar=False,vmax=max_n,
                    center=0)
        ax[idx].axis('off')
        
    for axs in ax:
        axs.axis('off')
    fig1.suptitle(plot_title)
    return fig1, fig2, fig3

def analyze_gratings(rois):
    
    a=1
    return rois

def find_edge_resp_decay(rois):
    """ For luminance edges to find decay from peak response"""
    for roi in rois:
        
        curr_trace = roi.edge_whole_traces_interpolated[roi.max_resp_idx-1][20:]
        max_resp = np.max(curr_trace)
        max_idx = np.argmax(curr_trace)
        last_resp = curr_trace[-1]
        
        decay_df = ((max_resp - last_resp) / max_resp)/(len(curr_trace)-max_idx) * 10
        roi.decay_strength = decay_df
    return rois
        

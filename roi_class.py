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
        
#%% Other functions     
        
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

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
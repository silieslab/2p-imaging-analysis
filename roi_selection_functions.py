# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:56:30 2021

@author: smolina and Burak Gur
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from roipoly import RoiPoly
from roi_class import ROI_bg
import copy
from tkinter import filedialog
import _pickle as cPickle


def select_regions(image_to_select_from, image_cmap ="gray",pause_t=7,
                   ask_name=True):
    """ Enables user to select rois from a given image using roipoly module.

    Parameters
    ==========
    image_to_select_from : numpy.ndarray
        An image to select ROIs from
    
    Returns
    =======
    
    """
    import warnings 
    plt.close('all')
    stopsignal = 0
    roi_number = 0
    roi_masks = []
    mask_names = []
    
    im_xDim = np.shape(image_to_select_from)[0]
    im_yDim = np.shape(image_to_select_from)[1]
    mask_agg = np.zeros(shape=(im_xDim,im_yDim))
    iROI = 0
    plt.style.use("dark_background")
    while (stopsignal==0):

        
        # Show the image
        fig = plt.figure()
        plt.imshow(image_to_select_from, interpolation='nearest', cmap=image_cmap)
        plt.colorbar()
        plt.imshow(mask_agg, alpha=0.3,cmap = 'tab20b')
        plt.title("Select ROI: ROI%d" % roi_number)
        plt.show(block=False)
       
        
        # Draw ROI
        curr_roi = RoiPoly(color='r', fig=fig)
        iROI = iROI + 1
        # plt.waitforbuttonpress()
        # plt.pause(pause_t)
        if ask_name:
            mask_name = input("\nEnter the ROI name:\n>> ") # Python 3.X
            
        else:
            mask_name = iROI
        curr_mask = curr_roi.get_mask(image_to_select_from)
        if len(np.where(curr_mask)[0]) ==0 :
            warnings.warn('ROI empty.. discarded.') 
            continue
        mask_names.append(mask_name)
        
        
        roi_masks.append(curr_mask)
        
        mask_agg[curr_mask] += 1
        
        
        
        roi_number += 1
        signal = input("\nPress k for exiting program, otherwise press enter") # Python 3.X
        
        if (signal == 'k\r'):
            stopsignal = 1
        elif (signal == 'k\\r'):
            stopsignal = 1
        elif (signal == 'k'):
            stopsignal = 1
        
    
    return roi_masks, mask_names


def generate_roi_masks_image(roi_masks,im_shape):
    # Generating an image with all clusters
    all_rois_image = np.zeros(shape=im_shape)
    all_rois_image[:] = np.nan
    for index, roi in enumerate(roi_masks):
        curr_mask = roi
        all_rois_image[curr_mask] = index + 1
    return all_rois_image

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
    

def run_roi_transfer(transfer_data_path, transfer_type,experiment_info=None,
                     imaging_info=None):
    '''
    
  
    '''
    load_path = open(transfer_data_path, 'rb')
    workspace = cPickle.load(load_path)
    rois = workspace['final_rois']
    
    if transfer_type == 'minimal' :
        print('Transfer type is minimal... Transferring just masks, categories and if present RF maps...\n')
        properties = ['category','analysis_params','RF_maps','RF_map',
                      'RF_center_coords','RF_map_norm']
        transferred_rois = transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    else:
        raise NameError('Invalid ROI transfer type')
        
        
   
    return transferred_rois




def run_ROI_selection(extraction_params, stack,stimulus_information, imaging_information, image_to_select=None):
    """
    THIS IS DOING THIS
    Parameters
    ==========
   
    XXXXXXXXXXX
        
    Returns
    =======
 
    XXXXXXXXXXX  
    

    """
    
    # Initialyze dictionary
    ROI_selection_dict = {}
    # Categories can be used to classify ROIs depending on their location
    # Backgroud mask (named "bg") will be used for background subtraction
    plt.close('all')
    plt.style.use("default")
    print('\n\nSelect categories and background')
    [cat_masks, cat_names] = select_regions(image_to_select, 
                                            image_cmap="viridis",
                                            pause_t=8)
    
    ROI_selection_dict['cat_masks'] = cat_masks
    ROI_selection_dict['cat_names'] = cat_names
    ROI_selection_dict['rois'] = None
    
    # have to do different actions depending on the extraction type
    if extraction_params['type'] == 'manual':
        print('\n\nSelect ROIs')
        [roi_masks, roi_names] = select_regions(image_to_select, 
                                                image_cmap="viridis",
                                                pause_t=4.5,
                                                ask_name=False)
        all_rois_image = generate_roi_masks_image(roi_masks,
                                                  np.shape(image_to_select))
        
        ROI_selection_dict['roi_masks'] = roi_masks
        ROI_selection_dict['roi_names'] = roi_names
        ROI_selection_dict['all_rois_image'] = all_rois_image
        
        return ROI_selection_dict
        #return cat_masks, cat_names, roi_masks, all_rois_image, None, None
            
    
    elif extraction_params['type'] == 'transfer':
        
        rois = run_roi_transfer(extraction_params['transfer_data_path'],
                                extraction_params['transfer_type'],
                                experiment_info=extraction_params['experiment_conditions'],
                                imaging_info=extraction_params['imaging_information'])
        
        ROI_selection_dict['rois'] = rois
        return ROI_selection_dict
        #return cat_masks, cat_names, None, None, rois, None
    
    #  Juan doing some magic here:
    elif extraction_params['type'] == 'cluster_analysis': 
        stimulus_information = stimulus_information
        imaging_information = imaging_information
    
    else:
       raise TypeError('ROI selection type not understood.') 
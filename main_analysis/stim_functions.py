# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:38:23 2021

@author: smolina and Burak Gur
"""
#%% Importing packages
import re
import os
import numpy as np
from itertools import islice
import glob


#%% Functions

def readStimOut(stimOutFile, skipHeader):
    """Read and get the stimulus output data.

    Parameters
    ==========
    stimOutFile : str
        Stimulus output file path.

    skipHeader : int, optional
        Default: 1

        Number of lines to be skipped from the beginning of the stimulus
        output file.

    Returns
    =======
    stimType : str
        Path of the executed stimulus file as it appears in the header of the
        stimulus output file.

    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.
    """
    # skip the first line since it is a file path
    rawStimData = np.genfromtxt(stimOutFile, dtype='float',
                                   skip_header=skipHeader,delimiter=',') # Seb: delimiter = ',' for _stimulus_ouput from 2pstim
    # also get the file path
    # do not mix it with numpy
    # only load and read the first line
    stimType = "stimType"
    with open(stimOutFile, 'r') as infile:
        lines_gen = islice(infile, 2) # Seb: the stimType is printed in line'2' of the _stimulus_ouput from 2pstim
        for line in lines_gen:
            line = re.sub('\n', '', line)   # get rid of 'line feed' (new line) -JC
            line = re.sub('\r', '', line)   # get rid of 'carriage return' (goes to beginning of new line, similar to \n) -JC
            line = re.sub(' ', '', line)    # get rid of space -JC
            stimType = line

    return stimType, rawStimData



def getEpochCount(rawStimData, epochColumn=3):
    """Get the total epoch number.

    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    Returns
    =======
    epochCount : int
        Total number of epochs.
    """
    # get the max epoch count from the rawStimData
    # 4th column is the epoch number
    # add plus 1 since the min epoch no is zero
    
    # BG edit: Changed the previous epoch extraction, which uses the maximum 
    # number + 1 as the epoch number, to a one finding the unique values and 
    # taking the length of it

    #JC: first get the shape (how many (rows, colums)) from the output file
    # then we specify which column we want the shape from (in the end just
    # #of rows from one column), here the epoch column.
    # With np.unique we can count how many differnt values would be in the epoch
    # column. And the last [0] is to get the first value of the tuple(row count)

    #same function is in epoch_functions -JC
    #they are not called from this file but only from the epoch_funcions one -JC

    epochCount = np.shape(np.unique(rawStimData[:, epochColumn]))[0]
    print("Number of epochs = " + str(epochCount))

    return epochCount


def readStimInformation(stimType, stimInputDir):
    """
    Parameters
    ==========
    stimType : str
        Path of the executed stimulus file as it appears in the header of the
        stimulus output file. *Required* if `gui` is FALSE and `stimInputDir`
        is given a non-default value.

    stimInputDir : str
        Path of the directory to look for the stimulus input.

    Returns
    =======
    stimInputFile : str
        Path to the stimulus input file (which contains stimulus parameters,
        not the output file).

    stimInputData : dict
        Lines of the stimulus generator file. Keys are the first terms of the
        line (the parameter names), values of the key is the rest of the line
        (a list).



    """

    stimType = stimType.split('/')[-1] # Seb: \\ to / 
    # only get last part of stimType, which is the name of stimInput file -JC

    try:
        stimInputFile = glob.glob(os.path.join(stimInputDir, stimType))[0]
    except IOError:
        print("Error: can\'t find file")
            
    stimInputData = {}
    with open(stimInputFile) as file:
        for line in file:
            curr_list = line.split()

            if not curr_list:
                 continue
                    
            key = curr_list.pop(0)  #remove zeros -JC
                
            if len(curr_list) == 1 and not "Stimulus." in key:
                try:
                    stimInputData[key] = int(curr_list[0])  #what to do if ture? -JC
                except ValueError:
                    stimInputData[key] = curr_list[0]
                continue 
                    
            if key.startswith("Stimulus."):
                key = key[9:]
                    
                if key.startswith("stimtype"):
                    stimInputData[key] = list(map(str, curr_list)) #only str in file -JC
                else: 
                    stimInputData[key] = list(map(float, curr_list))


    return stimInputFile, stimInputData

def divide_all_epochs(rawStimData, epochCount, framePeriod, trialDiff=0.20,
                      epochColumn=3, imgFrameColumn=7,checkLastTrialLen=True):
    """
    
    Finds all trial and epoch beginning and end frames
    
    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochCount : int
        Total number of epochs.

    framePeriod : float
        Time it takes to image a single frame.

    trialDiff : float
        Default: 0.20

        A safety measure to prevent last trial of an epoch being shorter than
        the intended trial duration, which can arise if the number of frames
        was miscalculated for the t-series while imaging. *Effective if and
        only if checkLastTrialLen is True*. The value is used in this way
        (see the corresponding line in the code):

        *(lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff*

        If the case above is satisfied, then this last trial is not taken into
        account for further calculations.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    imgFrameColumn : int, optional
        Default: 7

        The index of imaging frame column in the stimulus output file
        (start counting from 0).
        

    checkLastTrialLen :

    Returns
    =======
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list. Each term
        in this list is a trial of the epoch. These terms have the following
        structure: [[X, Y], [X, Y]] where X is the trial beginning and Y 
        is the trial end.

    trialCount : list
        Min (first term in the list) and Max (second term in the list) number
        of trials. Ideally, they are equal, but if the last trial is somehow
        discarded, e.g. because it ran for a shorter time period, min will be
        (max-1).
    """
    #same function is in epoch_functions -JC
    #they are not called from this file but only from the epoch_funcions one -JC

    trialDiff = float(trialDiff)
    trialCoor = {}
    
    for epoch in range(0, epochCount):
        
        trialCoor[epoch] = []   #key for each epoch -JC

    previous_epoch = []
    for line in rawStimData:
        
        current_epoch = int(line[epochColumn])      # defining which epoch we are in/ need to define beginning of new epoch -JC
        
        if (not(previous_epoch == current_epoch )): # Beginning of a new epoch trial
            
            
            if (not(previous_epoch==[])): # If this is after stim start (which is normal case)
                epoch_trial_end_frame = previous_frame  #previouse_frame is last frame from epoch, which is definded later -JC
                trialCoor[previous_epoch].append([[epoch_trial_start_frame, epoch_trial_end_frame], 
                                            [epoch_trial_start_frame, epoch_trial_end_frame]])
                                            #why do it 2 times the same? maybe to calculate the
                                            # trialCount (if last epoch was too short) -JC
                epoch_trial_start_frame = int(line[imgFrameColumn]) #define new start for current epoch -JC
                previous_epoch = int(line[epochColumn]) #epoch number of the current epoch to compare to next line -JC
                
            else:   # start of stimulation -JC
                previous_epoch = int(line[epochColumn])
                epoch_trial_start_frame = int(line[imgFrameColumn])
                
        previous_frame = int(line[imgFrameColumn])  #for each line define this variable new, untill the epoche changes, than this
                                                    #would show the last frame of the previouse epoch -JC
        
    if checkLastTrialLen:   #True or False -JC
        for epoch in trialCoor:
            delSwitch = False
            lenFirstTrial = (trialCoor[epoch][0][0][1]
                             - trialCoor[epoch][0][0][0])
            lenLastTrial = (trialCoor[epoch][-1][0][1]
                            - trialCoor[epoch][-1][0][0])
    
            if (lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff:
                delSwitch = True
    
            if delSwitch:
                print("Last trial of epoch " + str(epoch)
                      + " is discarded since the length was too short")
                trialCoor[epoch].pop(-1)
                
    trialCount = []
    epochTrial = []
    for epoch in trialCoor:
        epochTrial.append(len(trialCoor[epoch]))
    # in this case first element in trialCount is min no of trials
    # second element is the max no of trials
    trialCount.append(min(epochTrial))
    trialCount.append(max(epochTrial))
       

    return trialCoor, trialCount

# %%

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 17:25:19 2021

@author: smolina and Burak Gur
"""
#%% Importing packages
import numpy as np



#%% Functions
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
    epochCount = np.shape(np.unique(rawStimData[:, epochColumn]))[0]
    print("Number of epochs = " + str(epochCount))

    return epochCount

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
    trialDiff = float(trialDiff)
    trialCoor = {}
    
    for epoch in range(0, epochCount):
        
        trialCoor[epoch] = []

    previous_epoch = []
    for line in rawStimData:
        
        current_epoch = int(line[epochColumn])
        
        if (not(previous_epoch == current_epoch )): # Beginning of a new epoch trial
            
            
            if (not(previous_epoch==[])): # If this is after stim start (which is normal case)
                epoch_trial_end_frame = previous_frame
                trialCoor[previous_epoch].append([[epoch_trial_start_frame, epoch_trial_end_frame], 
                                            [epoch_trial_start_frame, epoch_trial_end_frame]])
                epoch_trial_start_frame = int(line[imgFrameColumn])
                previous_epoch = int(line[epochColumn])
                
            else:
                previous_epoch = int(line[epochColumn])
                epoch_trial_start_frame = int(line[imgFrameColumn])
                
        previous_frame = int(line[imgFrameColumn])
        
    if checkLastTrialLen:
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

def divideEpochs(rawStimData, epochCount, isRandom, framePeriod,
                 trialDiff=0.20, overlappingFrames=0, firstEpochIdx=0,
                 epochColumn=3, imgFrameColumn=7, incNextEpoch=True,
                 checkLastTrialLen=True):
    """
    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochCount : int
        Total number of epochs.

    isRandom : int

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

    overlappingFrames : int

    firstEpochIdx : int
        Default: 0

        Index of the first epoch.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    imgFrameColumn : int, optional
        Default: 7

        The index of imaging frame column in the stimulus output file
        (start counting from 0).

    incNextEpoch :
    checkLastTrialLen :

    Returns
    =======
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list. Each term
        in this list is a trial of the epoch. These terms have the following
        structure: [[X, Y], [Z, D]] where first term is the trial beginning
        (first of first) and end (second of first), and second term is the
        baseline start (first of second) and end (second of second) for that
        trial.

    trialCount : list
        Min (first term in the list) and Max (second term in the list) number
        of trials. Ideally, they are equal, but if the last trial is somehow
        discarded, e.g. because it ran for a shorter time period, min will be
        (max-1).

    isRandom :
    """
    trialDiff = float(trialDiff)
    firstEpochIdx = int(firstEpochIdx)
    overlappingFrames = int(overlappingFrames)
    trialCoor = {}
    fullEpochSeq = []

    if isRandom == 0:
        fullEpochSeq = range(epochCount)
        # if the key is zero, that means isRandom is 0
        # this is for compatibitibility and
        # to make unified workflow with isRandom == 1
        trialCoor[0] = []

    elif isRandom == 1:
        # in this case fullEpochSeq is just a list of dummy values
        # important thing is its length
        # it's set to 3 since trials will be sth like: 0, X
        # if incNextEpoch is True, then it will be like : 0,X,0
        fullEpochSeq = range(2)
        for epoch in range(1, epochCount):
            # add epoch numbers to the dictionary
            # do not add the first epoch there
            # since it is not the exp epoch
            # instead it is used for baseline and inc coordinates
            trialCoor[epoch] = []

    if incNextEpoch:
        # add the first epoch
        fullEpochSeq.append(firstEpochIdx)
    elif not incNextEpoch:
        pass

    # min and max img frame numbers for each and every trial
    # first terms in frameBaselineCoor are the trial beginning and end
    # second terms are the baseline start and end for that trial
    currentEpochSeq = []
    frameBaselineCoor = [[0, 0], [0, 0]]
    nextMin = 0
    baselineMax = 0

    for line in rawStimData:
        if (len(currentEpochSeq) == 0 and
                len(currentEpochSeq) < len(fullEpochSeq)):
            # it means it is the very beginning of a trial block.
            # in the very first trial,
            # min frame coordinate cannot be set by nextMin.
            # this condition satisfies this purpose.
            currentEpochSeq.append(int(line[epochColumn]))
            if frameBaselineCoor[0][0] == 0:
                frameBaselineCoor[0][0] = int(line[imgFrameColumn])
                frameBaselineCoor[1][0] = int(line[imgFrameColumn])

        elif (len(currentEpochSeq) != 0 and
              len(currentEpochSeq) < len(fullEpochSeq)):
            # only update the current epoch list
            # already got the min coordinate of the trial
            if int(line[epochColumn]) != currentEpochSeq[-1]:
                currentEpochSeq.append(int(line[epochColumn]))

            elif int(line[epochColumn]) == currentEpochSeq[-1]:
                if int(line[epochColumn]) == 0 and currentEpochSeq[-1] == 0:
                    # set the maximum endpoint of the baseline
                    # for the very first trial
                    frameBaselineCoor[1][1] = (int(line[imgFrameColumn])
                                               - overlappingFrames)

        elif len(currentEpochSeq) == len(fullEpochSeq):
            if nextMin == 0:
                nextMin = int(line[imgFrameColumn]) + overlappingFrames

            if int(line[epochColumn]) != currentEpochSeq[-1]:
                currentEpochSeq.append(int(line[epochColumn]))

            elif int(line[epochColumn]) == currentEpochSeq[-1]:
                if int(line[epochColumn]) == 0 and currentEpochSeq[-1] == 0:
                    # set the maximum endpoint of the baseline
                    # for all the trials except the very first trial
                    baselineMax = int(line[imgFrameColumn]) - overlappingFrames

        else:
            frameBaselineCoor[0][1] = (int(line[imgFrameColumn])
                                       - overlappingFrames)

            if frameBaselineCoor[0][1] > 0:
                if isRandom == 0:
                    # if the key is zero, that means isRandom is 0
                    # this is for compatibitibility and
                    # to make unified workflow isRandom == 1
                    trialCoor[0].append(frameBaselineCoor)
                elif isRandom == 1:
                    # get the epoch number
                    # epoch no should be the 2nd term in currentEpochSeq
                    expEpoch = currentEpochSeq[1]
                    trialCoor[expEpoch].append(frameBaselineCoor)
            # this is just a safety check
            # towards the end of the file, the number of epochs might
            # not be enough to form a trial block
            # so if the max img frame coordinate is still 0
            # it means this not-so-complete trial will be discarded
            # only complete trials are appended to trial coordinates
            # if it has a max frame coord, it is safe to say
            # it had nextMin in frameBaselineCoor
            # print(currentEpochSeq)
            currentEpochSeq = []
            currentEpochSeq.append(firstEpochIdx)
            # each time currentEpochSeq resets means that
            # one trial block is complete
            # adding firstEpochIdx is necessary
            # otherwise currentEpochSeq will shift by 1
            # after every trial cycle
            # now that the frame coordinates are stored
            # can reset it
            # and add min coordinate for the next trial
            # then add the max baseline coordinate for the next trial
            frameBaselineCoor = [[0, 0], [0, 0]]
            frameBaselineCoor[0][0] = nextMin
            frameBaselineCoor[1][0] = nextMin
            frameBaselineCoor[1][1] = baselineMax
            nextMin = 0
            baselineMax = 0

    # @TODO: no need to separate isRandoms, make a unified for loop
    if checkLastTrialLen:
        if isRandom == 0:
            lenFirstTrial = trialCoor[0][0][0][1] - trialCoor[0][0][0][0]
            lenLastTrial = trialCoor[0][-1][0][1] - trialCoor[0][-1][0][0]
            if ((lenFirstTrial - lenLastTrial) * framePeriod) >= trialDiff:
                trialCoor[0].pop(-1)
                print("Last trial is discarded since the length was too short")

        elif isRandom == 1:
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
    if isRandom == 0:
        # there is only a single key in the trialCoor dict in this case
        trialCount.append(len(trialCoor[0]))
    elif isRandom == 1:
        epochTrial = []
        for epoch in trialCoor:
            epochTrial.append(len(trialCoor[epoch]))
        # in this case first element in trialCount is min no of trials
        # second element is the max no of trials
        trialCount.append(min(epochTrial))
        trialCount.append(max(epochTrial))

    return trialCoor, trialCount, isRandom
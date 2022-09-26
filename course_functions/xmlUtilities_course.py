#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:13:12 2018

@author: motoko and Burak Gur
"""

from __future__ import division
import xml.etree.ElementTree as ET
import numpy


def getMicRelativeTime(xmlFile):
    """ Gets the relative microscope times from the XML output.

    Parameters
    ==========
    xmlFile : str
        XML file path.

    Returns
    =======
    micRelTimes : ndarray
    """
    # @TODO check if tehre is a way to load the xml file as columns and rows
    # so that you can get rid of the for loop and just slice a column
    # get relative time points of the microscope xml file
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    micRelTimes = numpy.array([])
    for frame in root.iter('Frame'):
        timept = float(frame.attrib['relativeTime'])
        micRelTimes = numpy.append(micRelTimes, timept)

    return micRelTimes


def getFramePeriod(xmlFile):
    """Gets the frame period from the XML file.

    Parameters
    ==========
    xmlFile : str
        XML file path.

    Returns
    =======
    framePeriod : float
    """
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    framePeriod = None
    for stateVal in root.iter('PVStateValue'):
        if stateVal.get('key') == 'framePeriod':
            framePeriod = float(stateVal.get('value'))

    return framePeriod


def getPixelSize(xmlFile):
    """Gets the frame period from the XML file.

    Parameters
    ==========
    xmlFile : str
        XML file path.

    Returns
    =======
    framePeriod : float
    """
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    for stateVal in root.iter('PVStateValue'):
        if stateVal.get('key') == 'micronsPerPixel':
            for idxValues in stateVal:
                if idxValues.get('index') == "XAxis":
                    x_size = float(idxValues.get('value'))
                elif idxValues.get('index') == "YAxis":
                    y_size = float(idxValues.get('value'))
                    
    pixelArea = x_size*y_size # in micron square
                    
    return x_size, y_size, pixelArea



def getLayerPosition(xmlFile):
    """Gets the layer postions from the xml file.

    Parameters
    ==========
    xmlFile : str
        XML file path.

    Returns
    =======
    layerPosition : list
        Positions as [X, Y, Z]
    """
    # @TODO might want to find a shorter way without going into loops
    # but it is super fast even that way, so not a priority
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    # positions are [x,y,z], respectively
    layerPosition = [0, 0, 0]
    for stateVal in root.iter('PVStateValue'):
        if stateVal.get('key') == 'positionCurrent':
            for idxValues in stateVal:
                if idxValues.get('index') == "XAxis":
                    for subIdx in idxValues:
                        if subIdx.get('subindex') == '0':
                            layerPosition[0] = float(subIdx.get('value'))
                elif idxValues.get('index') == "YAxis":
                    for subIdx in idxValues:
                        if subIdx.get('subindex') == '0':
                            layerPosition[1] = float(subIdx.get('value'))
                elif idxValues.get('index') == "ZAxis":
                    for subIdx in idxValues:
                        if subIdx.get('subindex') == '0':
                            layerPosition[2] = float(subIdx.get('value'))

    return layerPosition


    
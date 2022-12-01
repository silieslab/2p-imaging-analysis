# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 13:05:20 2021

@authors: Burak Gur, Sebastian Molina-Obando
"""

import os
import re
import numpy as np
import _pickle as cPickle # For Python 3.X

def makePath(path):
    """Make Windows and POSIX compatible absolute paths automatically.

    Parameters
    ==========
    path : str

    Path to be converted into Windows or POSIX style path.

    Returns
    =======
    compatPath : str
    """

    compatPath = os.path.abspath(os.path.expanduser(path))

    return compatPath

def getVarNames(varFile='variablesToSave.txt'):
    """ Read the variable names from a plain-text document. Then it is used to
    save and load the variables by conserving the variable names, in other
    functions. Whenever a new function is added, one should also add the stuff
    it returns (assuming returned values are stored in the same variable names
    as in the function definition) to the varFile.

    Parameters
    ==========
    varFile : str, optional
        Default: 'variablesToSave.txt'

        Plain-text file from where variable names are read.

    Returns
    =======
    varNames : list
        List of variable names
    """
    # get the variable names
    varFile = makePath(varFile)
    workspaceVar = open(varFile, 'r')
    varNames = []

    for line in workspaceVar:
        if line.startswith('#'):
            pass
        else:
            line = re.sub('\n', '', line)
            line = re.sub('', '', line)
            line = re.sub(' ', '', line)
            if line == '':
                pass
            else:
                varNames.append(line)
    workspaceVar.close()

    return varNames


def saveWorkspace(outDir, baseName, varDict, varFile='workspaceVar.txt',
                  extension='.pickle'):
    """ Save the variables that are present in the varFile. The file format is
    Pickle, which is a mainstream python format.

    Parameters
    ==========
    outDir : str
        Output diectory path.

    baseName : str
        Name of the time series folder.

    varDict : dict

    varFile : str, optional
        Default: 'workspaceVar.txt'

        Plain-text file from where variable names are read.

    extension : str, optional
        Default: '.pickle'

        Extension of the file to be saved.

    Returns
    =======
    savePath : str
        Path (inc. the filename) where the analysis output is saved.
    """

    # it is safer to get the variables from a txt
    # otherwise the actual session might have some variables
    # @TODO make workspaceFl path not-hardcoded
    print(varFile)
    varFile = makePath(varFile)
    varNames = getVarNames(varFile=varFile)
    workspaceDict = {}

    for variable in varNames:
        try:
            # only get the wanted var names from globals
            workspaceDict[variable] = varDict[variable]
        except KeyError:
            pass

    # open in binary mode and use highest cPickle protocol
    # negative protocol means highest protocol: faster
    # use cPickle instead of pickle: faster
    # C implementation of pickle
    savePath = os.path.join(outDir, baseName + extension)
    saveVar = open(savePath, "wb")
    cPickle.dump(workspaceDict, saveVar, protocol=-1)


    
    saveVar.close()

    return savePath
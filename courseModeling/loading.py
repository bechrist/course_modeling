# %% load.py - Data selection & loading operations


import os

from typing import Any
import numpy.typing as npt
from courseModeling.course import Course

import pickle
import cv2 as cv




def loader(directory: str, input: str) -> npt.ArrayLike | Course: 
    # Determine input file location
    inputPath = get_input_path(directory, input)

    if inputPath == None:
        raise ValueError('Input file not found within specified data directory')
    elif inputPath.split('.')[1] in ['.pkl', '.pickle']:
        data = pickle_reader(inputPath)
    else:   
        data = image_reader(inputPath)

    return data


def get_input_path(directory: str, input: str) -> str:
    for root, dirs, files in os.walk(directory):
        if len(files) == 0: continue

        testPath = os.path.abspath(os.path.join(root,input))
        if os.path.isfile(testPath):
            return testPath
    
    return None


def image_reader(path: str) -> npt.ArrayLike: 
    return cv.imread(path)


def pickle_reader(path: str) -> Any:
    with open(path, 'r'):
        return pickle.load(path)
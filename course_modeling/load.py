"""load.py - Data Loading and Parsing"""
import numpy.typing as npt

import re

import numpy as np

import pickle as pkl
import json
import matplotlib as mpl

import os, sys;

sys.path.insert(0, os.path.realpath('.'))
from course_modeling.course import Course

__all__ = ['load_data_file', 
           'read_pickle', 'read_json', 'read_image',
           'get_input_path', 'get_bool']

def load_data_file(input: str, directory: str) -> npt.ArrayLike: 
    input_path = get_input_path(input, directory)
    if input_path == None:
        raise ValueError('Input file not found within specified data directory')
    
    ext = input_path.split('.')[1].lower()
    match ext:
        case 'pkl' | 'pickle':
            data = read_pickle(input_path)
        case 'json':
            data = read_web_plot_digitizer_json(input_path)
        case 'png' | 'jpg' | 'jpeg':   
            data = read_image(input_path)
        case _:
            raise ValueError('Input file extension not recognized')

    return data

# %% Readers
def read_pickle(path: str) -> Course:
    """Read pre-existing course object"""
    with open(path, 'r'):
        return pkl.load(path)
    
def read_web_plot_digitizer_json(path: str) -> Course:
    """Read WebPlotDigitizer project data"""
    # Load project datasets
    with open(path, 'r') as file:
        project = json.load(file)
    
    dataset = {ds['name']: ds for ds in project['datasetColl']}

    # Loop 
    loop = False
    if 'endurance' in path:
        loop = True
    else:
        loop = get_bool("Is the course a loop?")

    # Scaling
    scale_key = next(key for key in dataset.keys() if 'scaling' in key.lower())
    scale_str = scale_key.split('(')[1].split(')')[0]
    
    num_regex = re.compile("([0-9]+)")
    scale_dist_str = num_regex.match(scale_str).groups()[0]
    
    scale_unit = scale_str[len(scale_dist_str):]
    match scale_unit: 
        case "m":
            scale_dist = float(scale_dist_str)
        case "'" | "ft": 
            scale_dist = float(scale_dist_str) / 3.28084
        case _:
            raise ValueError(("Scale unit not recognized, please use meters (m) "
                              "or feet (ft or ')"))
        
    scale_points = np.array([point['value'] for point in dataset[scale_key]['data']])
    scale_length = np.linalg.norm(np.diff(scale_points, axis=0))

    scale_factor = scale_dist / scale_length # [m/px]

    # Boundary
    boundary = [[], []]
    for name, ds in filter(lambda item: 'boundary' in item[0].lower(), dataset.items()):
        if 'left' in name.lower():
            j = 0
        elif 'right' in name.lower():
            j = 1
        else:
            raise ValueError('Boundary must be labeled as left of right')
        
        boundary[j] = np.array([point['value'] for point in ds['data']]).T

    # Image data
    wpd_folder, data_file = os.path.split(path)
    data_folder = os.path.dirname(wpd_folder)

    image_file = os.path.splitext(data_file)[0] + '.png' 
    image_path = get_input_path(image_file, data_folder)
    image = read_image(image_path)

    return Course(image_path, image, boundary, [], [], scale_factor, loop)

def read_image(path: str) -> npt.ArrayLike: 
    return mpl.image.imread(path)

# %% Helpers
def get_input_path(input: str, directory: str) -> str | None:
    for root, _, files in os.walk(directory):
        if len(files) == 0: continue

        test_path = os.path.abspath(os.path.join(root,input))
        if os.path.isfile(test_path):
            return test_path
    
    return None

def get_bool(prompt: str) -> bool:
    while True:
        try:
            return {'t': True, 'true': True, 'f': False, 'false': False}[input(prompt).lower()]
        except KeyError:
            print("Invalid input, please enter True (T) or False (F)")
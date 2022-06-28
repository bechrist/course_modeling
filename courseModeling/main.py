# %% main.py - Executes Main Course Modeling Pipeline

import os

from courseModeling.course import CourseBuilder


# %% Parameters
SRC_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(SRC_DIR,'..','data'))

INPUT = '16_19_lincoln_endurance.png'

# %% Initialize Course Builder
builder = CourseBuilder(INPUT,DATA_DIR)

# %% Course Data Selection 


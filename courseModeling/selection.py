# %% selection.py - Boundary & Scale Selection Interactive Drawing GUI Tool


import cv2 as cv
import numpy as np

from enum import Enum, auto
import numpy.typing as npt


class ROI(Enum):
    LINE = auto()
    RECTANGLE = auto()
    SQUARE = auto()
    ELLIPSE = auto()
    CIRCLE = auto()
    POLYLINE = auto()

events = [i for i in dir(cv) if 'EVENT' in i]
print( events )

class drawingGUI:
    def __init__(self, img: npt.ArrayLike = None):
        self.img = np.full()


    
def draw_polyline(event,x,y,flag,param):
    if event == cv.EVENT_LBUTTONDOWN:
        _drawing = True
    elif event == cv.EVENT_MOUSEMOVE:
        if _drawing == True:
            a = 1

    
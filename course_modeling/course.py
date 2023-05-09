"""course.py - Course Definition"""
import typing as typ
from dataclasses import dataclass, field

import numpy as np

import matplotlib.pyplot as plt

__all__ = ['Course', 'Trajectory']

@dataclass(slots=True)
class Course:
    """Course layout data"""
    image_path: str = None
    image: np.ndarray = np.array([])
    boundary: tuple[np.ndarray, np.ndarray] = field(default_factory=lambda: (np.array([]), np.array([])))
    post: tuple[np.ndarray, np.ndarray] = field(default_factory=lambda: (np.array([]), np.array([])))
    gate: np.ndarray = field(default_factory=lambda: np.array([]))
    scale_factor: float = None      # [m/px]                              
    loop: bool = None

    def plot(self):
        fig = plt.figure()
        #fig.set_dpi(150)

        if self.image is not None:
            plt.imshow(self.image)
        
        scale = 10 / np.max(fig.get_size_inches())
        fig.set_size_inches(fig.get_size_inches()*scale)

        if np.any(self.gate):
            pt = lambda i,j,k: self.post[i][j,self.gate[i,k]]
            for k in range(self.gate.shape[-1]):
                plt.plot([pt(0,0,k), pt(1,0,k)], [pt(0,1,k), pt(1,1,k)], 'k')
            
            plt.plot(self.post[0][0], self.post[0][1], 'k.')
            plt.plot(self.post[1][0], self.post[1][1], 'k.')

        if np.any(self.boundary[0]) or np.any(self.boundary[1]): 
            plt.plot(self.boundary[0][0], self.boundary[0][1], 'r.')
            plt.plot(self.boundary[1][0], self.boundary[1][1], 'b.')

        plt.axis('off')
        plt.show()

@dataclass(slots=True)
class Trajectory:
    course: Course
    weight: np.ndarray
    _arc_length: np.ndarray

    def eval(s: float) -> np.ndarray:
        """Given arc length return position and heading angle"""
        raise NotImplementedError
    
    def curvature(s: float) -> float:
        """Evaluate curvature at a point on the spline"""
        raise NotImplementedError
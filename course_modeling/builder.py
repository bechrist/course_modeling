"""builder.py - CourseBuilder Definition"""
import numpy.typing as npt

import os, sys

sys.path.insert(0, os.path.realpath('.'))
from course_modeling.course import Course
from course_modeling.load import load_data_file
from course_modeling.gate import nearest_neighbor_gate

__all__ = ['CourseBuilder', 'TrajectoryBuilder']

class CourseBuilder:
    """Course object builder"""
    def __init__(self, input: str | Course, directory: str):
        """Initializes CourseBuilder"""
        self.reset()

        if isinstance(input, str):
            self.load(input, directory)
        elif isinstance(input, Course):
            self._course = input
        else:
            raise ValueError(('CourseBuilder may be given a path to an image or'
                              'dumped Course object, or an existing Course'
                              'object'))

    def reset(self):
        """Initizalize Course property"""
        self._course = Course()

    @property
    def course(self) -> Course:
        """Return existing Course property"""
        return self._course

    def load(self, input: str, directory: str):
        """Load data from file"""
        data = load_data_file(input, directory)

        if isinstance(data, Course):
            self._course = data
        elif isinstance(data, npt.ArrayLike) and len(data.shape) == 3:
            self._course.image = data
        else:
            raise ValueError(('Input data must be an image ArrayLike or a '
                              'Course object'))

    def scale(self):
        """Select scaling points and set scale distance"""
        raise NotImplementedError
    
    def boundary(self):
        """Select boundary data"""
        raise NotImplementedError
    
    def gates(self, spacing: float = 10.0):
        """Generate gates with set maximum spacing"""
        self._course.post, self._course.gate = nearest_neighbor_gate(
            self._course.boundary, self._course.scale_factor, spacing, self._course.loop)
# %% course.py - Course and Builder Class Definitions 


import numpy.typing as npt
from dataclasses import dataclass, field

from courseModeling.loading import loader
from courseModeling.selection import drawingGUI


@dataclass(slots=True)
class Course:
    """Describes race course layout"""
    image_path: str = None
    image: npt.ArrayLike = None
    boundary: list[npt.ArrayLike] = field(default_factory=list)
    gate: list[npt.ArrayLike] = field(default_factory=list)
    scale_len : float = None                             # Pixel Length [px]
    scale_dist: float = None                             # Physical Distance [m]
    loop: bool = None
    

class CourseBuilder:
    def __init__(self, input: str | Course, directory: str = None) -> None:
        self.reset()

        if isinstance(input, str):
            self.load(directory, input)
        elif isinstance(input, Course):
            self._course = input
        else:
            raise ValueError('CourseBuilder may be given a path to an image or dumped Course object, or an existing Course object')

    def reset(self) -> None:
        self._course = Course()

    @property
    def course(self) -> Course:
        course = self._course
        self.reset()
        return course

    def load(self, directory: str, input: str) -> None:
        data = loader(directory, input)

        if isinstance(data, Course):
            self._course = data
        elif isinstance(data, npt.ArrayLike) and len(data.shape) == 3:
            self._course.image = data
        else:
            raise ValueError('Input data must be an image ArrayLike or a Course object')

    def select(self) -> None:
        a = 1
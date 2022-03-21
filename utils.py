"""
Contains utilities, types, classes etc. used by the maths_images package

Written by: Benjamin Smith
     email: ben297@gmail.com
    github: https://github.com/obbiwan
"""
from typing import TypedDict

import numpy as np
import numpy.typing


class PlotData(TypedDict):
    """Defines the structure of a dictionary containing plotting data."""
    x: np.typing.NDArray
    y: np.typing.NDArray
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    plot_title: str
    filename: str

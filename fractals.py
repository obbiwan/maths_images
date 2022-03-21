"""
Contains functions which implement a variety of Fractals

Written by: Benjamin Smith
     email: ben297@gmail.com
    github: https://github.com/obbiwan
"""
import math
import random

import numba
import numpy as np

from utils import PlotData


def barnsley(n_points: int = 10**6) -> PlotData:
    """
    Calculates a list of (x, y) points according to Barnsley's Fern
    https://en.wikipedia.org/wiki/Barnsley_fern
    :param n_points: The number of (x, y) plot points to calculate
    :return: a PlotData dictionary, containing the data necessary to plot the Duffing Map
    """

    @numba.jit
    def rule(p):
        """Applies the Barnsley's Fern transformation to an (x, y) point to get the next (x, y) point"""
        r = random.random()
        if r < 0.01:
            return +0.00*p[0] + 0.00*p[1], +0.00*p[0] + 0.16*p[1] + 0.00
        elif r < 0.86:
            return +0.85*p[0] + 0.04*p[1], -0.04*p[0] + 0.85*p[1] + 1.60
        elif r < 0.93:
            return +0.20*p[0] - 0.26*p[1], +0.23*p[0] + 0.22*p[1] + 1.60
        else:
            return -0.15*p[0] + 0.28*p[1], +0.26*p[0] + 0.24*p[1] + 0.44

    # numpy arrays to store plot points
    x = np.zeros(n_points)
    y = np.zeros(n_points)

    # Set the starting point
    x[0], y[0] = 0.0, 0.0

    # Calculate the points
    for i in range(n_points-1):
        x[i+1], y[i+1] = rule((x[i], y[i]))

    # Calculate the plot area
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    return PlotData(x=x, y=y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, plot_title="Barnsley's Fern",
                    filename='barnsley')


def mandelbrot(x_min: float = -1.5, x_max: float = 0.5, y_min: float = -1.0, y_max: float = 1.0,
               samples: int = 10**6, max_iterations: float = 100) -> PlotData:
    """
    Calculates a list of (x, y) points according to the Mandelbrot Set
    https://en.wikipedia.org/wiki/Mandelbrot_set
    :param x_min: The lower bound of the horizontal axis
    :param x_max: The upper bound of the horizontal axis
    :param y_min: The lower bound of the vertical axis
    :param y_max: The upper bound of the vertical axis
    :param samples: The number of samples points to test from the plot area
    :param max_iterations: The maximum number of iterations to apply to each point before deciding it does not diverge
    :return: a PlotData dictionary, containing the data necessary to plot the Logistic Map
    """

    n_x = math.floor(math.sqrt(samples))
    n_y = math.floor(math.sqrt(samples))
    p_x, p_y = [], []

    @numba.jit
    def diverges(a: float, b: float):
        """Repeatedly applies the Mandelbrot transformation to a point and returns whether the point diverged"""
        z = 0.0
        i = 0
        while i < max_iterations:
            z = z ** 2 + (a + b * 1j)
            i += 1
            if abs(z) >= 2.0:
                return True
        return False

    for x in np.linspace(x_min, x_max, n_x):
        for y in np.linspace(y_min, y_max, n_y):

            if not diverges(x, y):
                p_x.append(x)
                p_y.append(y)

    return PlotData(x=p_x, y=p_y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, plot_title='Mandelbrot Set',
                    filename='mandelbrot')

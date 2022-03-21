"""
Contains functions which implement a variety of 2D Chaotic Maps

Written by: Benjamin Smith
     email: ben297@gmail.com
    github: https://github.com/obbiwan
"""
import math

import numba
import numpy as np

from utils import PlotData


def bogdanov(x_min: float = -1.0, x_max: float = 1.5, y_min: float = -1.0, y_max: float = 1.0,
             starting_points: int = 10**4, iterations_per_point: int = 500, skip_iterations: int = 100,
             e: float = 0.0, k: float = 1.2, u: float = 0.0) -> PlotData:
    """
    Calculates a list of (x, y) points according to the Bogdanov Map
    https://en.wikipedia.org/wiki/Bogdanov_map
    :param x_min: The lower bound of the horizontal axis
    :param x_max: The upper bound of the horizontal axis
    :param y_min: The lower bound of the vertical axis
    :param y_max: The upper bound of the vertical axis
    :param starting_points: The number of (x, y) points within the plot area from which to iterate; The chosen
                            starting points are equally-spaced over the plot area
    :param iterations_per_point: The number of times the mapping function will be applied from each starting point
    :param skip_iterations: Skips plotting the earliest iterations from each starting point to produce a cleaner image;
                            this is how many iterations from each starting point are omitted from the plot
    :param e: The 'epsilon' parameter of the Bogdanov map
    :param k: The 'kappa' parameter of the Bogdanov map
    :param u: The 'mu' parameter of the Bogdanov map
    :return: a PlotData dictionary, containing the data necessary to plot the Bogdanov Map
    """

    if skip_iterations >= iterations_per_point:
        raise ValueError('skip_iterations must be less than iterations_per_point')

    @numba.jit
    def bogdanov_rule(p):
        """Applies the Bogdanov map to an (x, y) point to get the next (x, y) point"""
        return p[0] + p[1] * (1.0 + e) + k * p[0] * (p[0] - 1.0) + u * p[0] * p[1], \
               p[1] * (1.0 + e) + k * p[0] * (p[0] - 1.0) + u * p[0] * p[1]

    # Splits the total number of starting points into an even grid if n_x * n_y points
    n_x = n_y = math.floor(math.sqrt(starting_points))

    # Determines the total number of points that get plotted
    n_points = (iterations_per_point - skip_iterations) * n_x * n_y

    # Ensures that numpy warnings are raised as errors, so that overflows can be caught and handled
    np.seterr(all='raise')

    # numpy arrays to store plot points
    x = np.zeros(n_points)
    y = np.zeros(n_points)

    i = 0
    for x_0 in np.linspace(x_min, x_max, num=n_x):
        for y_0 in np.linspace(y_min, y_max, num=n_y):

            start_x, start_y = x_0, y_0

            # Removes noise by not plotting the earliest iterations on each starting point
            for iteration in range(skip_iterations):
                try:
                    start_x, start_y = bogdanov_rule((start_x, start_y))
                # if overflow occurs, stop iterating on the current point
                except FloatingPointError:
                    break

            x[i], y[i] = start_x, start_y
            for iteration in range(iterations_per_point - skip_iterations - 1):
                try:
                    x[i+1], y[i+1] = bogdanov_rule((x[i], y[i]))
                    i += 1
                # if overflow occurs, stop iterating on the current point
                except FloatingPointError:
                    break

    return PlotData(x=x, y=y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, plot_title='Bogdanov Map',
                    filename='bogdanov')


def duffing(n_points: int = 10**6, x_0: float = 0.1, y_0: float = 0.1, a: float = 2.75, b: float = 0.15) -> PlotData:
    """
    Calculates a list of (x, y) points according to the Duffing Map
    https://en.wikipedia.org/wiki/Duffing_map
    :param n_points: The number of (x, y) plot points to calculate
    :param x_0: The horizontal coordinate of the point from which iteration starts
    :param y_0: The vertical coordinate of the point from which iteration starts
    :param a: The 'a' parameter of the Duffing map
    :param b: The 'b' parameter of the Duffing map
    :return: a PlotData dictionary, containing the data necessary to plot the Duffing Map
    """

    @numba.jit
    def rule(p):
        """Applies the Duffing map to an (x, y) point to get the next (x, y) point"""
        return p[1], -b*p[0] + a*p[1] - p[1]**3

    # numpy arrays to store plot points
    x = np.zeros(n_points)
    y = np.zeros(n_points)

    # Set the starting point
    x[0], y[0] = x_0, y_0

    # Calculate the points
    for i in range(n_points-1):
        x[i+1], y[i+1] = rule((x[i], y[i]))

    # Calculate the plot area
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    return PlotData(x=x, y=y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, plot_title='Duffing Map',
                    filename='duffing')


def gingerbreadman(n_points: int = 10**6, x_0: float = 1.5, y_0: float = 2.6) -> PlotData:
    """
    Calculates a list of (x, y) points according to the Gingerbreadman Map
    https://en.wikipedia.org/wiki/Gingerbreadman_map
    :param n_points:  The number of (x, y) plot points to calculate
    :param x_0: The horizontal coordinate of the point from which iteration starts
    :param y_0: The vertical coordinate of the point from which iteration starts
    :return: a PlotData dictionary, containing the data necessary to plot the Gingerbreadman Map
    """

    @numba.jit
    def rule(p):
        """Applies the Gingerbreadman map to an (x, y) point to get the next (x, y) point"""
        return 1 - p[1] + abs(p[0]), p[0]

    # numpy arrays to store plot points
    x = np.zeros(n_points)
    y = np.zeros(n_points)

    # Set the starting point
    x[0], y[0] = x_0, y_0

    # Calculate the points
    for i in range(n_points-1):
        x[i+1], y[i+1] = rule((x[i], y[i]))

    # Calculate the plot area
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    return PlotData(x=x, y=y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, plot_title='Gingerbreadman Map',
                    filename='gingerbreadman')


def henon(n_points: int = 10**6, x_0: float = 0.1, y_0: float = 0.2, a: float = 1.4, b: float = 0.3) -> PlotData:
    """
    Calculates a list of (x, y) points according to the Henon Map
    https://en.wikipedia.org/wiki/H%C3%A9non_map
    :param n_points: The number of (x, y) plot points to calculate
    :param x_0: The horizontal coordinate of the point from which iteration starts
    :param y_0: The vertical coordinate of the point from which iteration starts
    :param a: The 'a' parameter of the Henon map
    :param b: The 'b' parameter of the Henon map
    :return: a PlotData dictionary, containing the data necessary to plot the Henon Map
    """

    @numba.jit
    def rule(p):
        """Applies the Henon map to an (x, y) point to get the next (x, y) point"""
        return 1 - a*p[0]**2 + p[1], b*p[0]

    # numpy arrays to store plot points
    x = np.zeros(n_points)
    y = np.zeros(n_points)

    # Set the starting point
    x[0], y[0] = x_0, y_0

    # Calculate the points
    for i in range(n_points-1):
        x[i+1], y[i+1] = rule((x[i], y[i]))

    # Calculate the plot area
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    return PlotData(x=x, y=y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, plot_title='Henon Map',
                    filename='henon')


def logistic(x_min: float = 0.0, x_max: float = 4.0, y_min: float = 0.0, y_max: float = 1.0, y_start: float = 0.1,
             x_samples: int = 2000, iterations: int = 100, y_samples: int = 2000) -> PlotData:
    """
    Calculates a list of (x, y) points according to the Logistic Map
    https://en.wikipedia.org/wiki/Logistic_map
    :param x_min: The lower bound of the horizontal axis
    :param x_max: The upper bound of the horizontal axis
    :param y_min: The lower bound of the vertical axis
    :param y_max: The upper bound of the vertical axis
    :param y_start: The starting vertical coordinate from which iteration begins
    :param x_samples: The number of samples values from the horizontal axis
    :param iterations: The number of iterations to apply to each starting point before plotting data
    :param y_samples: The number of samples points to plot for each horizontal-axis value
    :return: a PlotData dictionary, containing the data necessary to plot the Logistic Map
    """

    @numba.jit
    def rule(p):
        """Applies the Logistic map to an (x, y) point to get the next (x, y) point"""
        return p[0] * p[1] * (1 - p[1])

    # Determines the total number of points that get plotted
    n_points = x_samples * y_samples

    # numpy arrays to store plot points
    x = np.zeros(n_points)
    y = np.zeros(n_points)

    i = 0
    for x_0 in np.linspace(x_min, x_max, x_samples):

        # Skips past the initial iterations of each point to remove noise
        y_0 = y_start
        for iteration in range(iterations):
            y_0 = rule((x_0, y_0))

        for iteration in range(y_samples):
            y_0 = rule((x_0, y_0))
            x[i], y[i] = x_0, y_0
            i += 1

    return PlotData(x=x, y=y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, plot_title='Logistic Map',
                    filename='logistic')


def tinkerbell(n_points: int = 10**6, x_0: float = -0.72, y_0: float = -0.64, a: float = 0.9, b: float = -0.6013,
               c: float = 2.0, d: float = 0.50) -> PlotData:
    """
    Calculates a list of (x, y) points according to the Tinkerbell Map
    https://en.wikipedia.org/wiki/Tinkerbell_map
    :param n_points: The number of (x, y) plot points to calculate
    :param x_0: The horizontal coordinate of the point from which iteration starts
    :param y_0: The vertical coordinate of the point from which iteration starts
    :param a: The 'a' parameter of the Tinkerbell Map
    :param b: The 'b' parameter of the Tinkerbell Map
    :param c: The 'c' parameter of the Tinkerbell Map
    :param d: The 'd' parameter of the Tinkerbell Map
    :return: a PlotData dictionary, containing the data necessary to plot the Tinkerbell Map
    """

    @numba.jit
    def rule(p):
        """Applies the Tinkerbell map to an (x, y) point to get the next (x, y) point"""
        return p[0]**2 - p[1]**2 + a*p[0] + b*p[1], 2*p[0]*p[1] + c*p[0] + d*p[1]

    # numpy arrays to store plot points
    x = np.zeros(n_points)
    y = np.zeros(n_points)

    # Set the starting point
    x[0], y[0] = x_0, y_0

    # Calculate the points
    for i in range(n_points-1):
        x[i+1], y[i+1] = rule((x[i], y[i]))

    # Calculate the plot area
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    return PlotData(x=x, y=y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, plot_title='Tinkerbell Map',
                    filename='tinkerbell')

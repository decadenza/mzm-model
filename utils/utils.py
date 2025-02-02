import math
from collections.abc import Callable
import numpy as np


def round_up_to_power_of_two(n):
    """ Round a number to the nearest upper power of two. """
    exp = math.ceil(np.log2(n))
    return 2**exp


def square_wave(t, period):
    """ Return a square wave between 0 and 1. """
    return np.where(np.floor(2*(t % period)/period) == 1, 1, 0)

def sine_wave_with_random_delay(t, period):
    """ Return a sine wave of the given period with a random delay. """

    # Adding a random shift.
    # The model must work with any shift.
    deltaT = np.random.uniform(0, period)

    return np.sin(2*np.pi*(t-deltaT)/period)


def fit_parabola(points):
    """
    Fits a parabola to three given points.

    Args:
        points: An NumPy array of (x, y) point, with shape (N,2).

    Returns:
        A tuple (a, b, c) representing the coefficients x^2, x, 1 of the parabola, respectively.
    """

    x_coords, y_coords = np.split(points, 2, axis=1)
    x_coords = np.ravel(x_coords)
    y_coords = np.ravel(y_coords)
    A = np.vstack([x_coords**2, x_coords, np.ones(x_coords.size)]).T
    return np.linalg.solve(A, y_coords)


def parabola_estimate_peak(points):
    """
    Given three points, find the parabola and return the peak.
    """

    # Compute the parabola.
    a, b, c = fit_parabola(points)
    
    # Find the peak.
    peak_x = -b / (2*a)
    peak_y = a*peak_x**2 + b*peak_x + c
    
    return peak_x, peak_y
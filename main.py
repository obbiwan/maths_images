"""
Creates plots of a variety of 2D Chaotic Maps and Fractals

Written by: Benjamin Smith
     email: ben297@gmail.com
    github: https://github.com/obbiwan
"""
import chaotic_maps as c
import fractals as f
from plotting import scatter

# Chaotic Maps
scatter(c.bogdanov())
scatter(c.duffing())
scatter(c.gingerbreadman())
scatter(c.henon())
scatter(c.logistic())
scatter(c.tinkerbell())

# Fractals
scatter(f.barnsley())
scatter(f.mandelbrot())

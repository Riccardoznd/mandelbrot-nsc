"""
Mandelbrot Set Generator numba version

Author: Riccardo Zanda
Course: Numerical Scientific Computing 2026
"""
import matplotlib.pyplot as plt
import time
import statistics
from numba import njit

#define region boundaries
x_min=-2.0
x_max=1.0
y_min=-1.5
y_max=1.5

#define resolution
width=4096
height=4096
max_iter=100

#approach A

@njit
def mandelbrot_point_numba(c, max_iter):
    z = 0j  # Complex literal required for type inference (Slide 35)
    for n in range(max_iter):
        # Condition must be on one line (Slide 34)
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z * z + c
    return max_iter #to check whether everithing worked out correctly

def mandelbrot_hybrid(xmin, xmax, ymin, ymax, width, height, max_iter):
    #equally spaced vectors
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    # createsn 2D array
    result = np.zeros((height, width), dtype=np.int32)
    
   #outer loop still in python
    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            result[i, j] = mandelbrot_point_numba(c, max_iter)
    return result

#approach B (suggested)
@njit
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0
            while n < max_iter and z.real * z.real + z.imag * z.imag <= 4.0:
                z = z * z + c
                n += 1
            result[i, j] = n
    return result

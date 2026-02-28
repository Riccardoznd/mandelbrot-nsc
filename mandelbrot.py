"""
Mandelbrot Set Generator

Author: Riccardo Zanda
Course: Numerical Scientific Computing 2026
"""
import numpy as np
import time
import statistics
import matplotlib.pyplot as plt

#define region boundaries
x_min=-2.0
x_max=1.0
y_min=-1.5
y_max=1.5

#define resolution
width=1024
height=1024
max_iter=100

def my_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter):

    #evenly spaced vectors
    x=np.linspace(x_min,x_max,width)
    y=np.linspace(y_min,y_max,height)

    #creating 2D map of the coordinates
    X,Y=np.meshgrid(x,y)

    #make complex numbers
    C= X +1j *Y

    print(f"shape: {C.shape}") #1024,1024
    print(f"Type:{C.dtype}") #complex128

    #now i inizialize Z used for the current value and M for the the iteration counting map
    Z = np.zeros_like(C) #_like makes it so that Z is also a complex number array
    M = np.zeros(C.shape, dtype=int)

    #we still need to make a loop because the formula of the mandelbrot set is recursive
    max_iter=100

    for n in range(max_iter):
        z = np.abs(Z) <= 2 #z acts as a mask
        Z[z] = Z[z]**2 + C[z]
        
        # Increment iteration count for those points
        M[z] += 1
    return M 

#using now the fucntion
M = my_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter)


plt.figure(figsize=(10, 10))
plt.imshow(M, extent=[-2, 1, -1.5, 1.5], origin='lower', cmap='hot')
plt.colorbar()
plt.show()

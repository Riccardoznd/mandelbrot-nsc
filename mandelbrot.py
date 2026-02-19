"""
Mandelbrot Set Generator

Author: Riccardo Zanda
Course: Numerical Scientific Computing 2026
"""
import numpy as np

#define region boundaries
x_min=-2.0
x_max=1.0
y_min=-1.5
y_max=1.5

#define resolution
width=1024
height=1024

#evenly spaced vectors
x=np.linspace(x_min,x_max,width)
y=np.linspace(y_min,y_max,height)

#creating 2D map of the coordinates
X,Y=np.meshgrid(x,y)

#make complex numbers
c= X +1j *Y

#I define the maximum interaction
max_iter=100
#i want to loop through the whole grid

for  i in range(height):
for j in range (widht):
z=0
C=c[i,j]
for n in range (0,max_iter):
z=z*z+c
print(z)


    pass
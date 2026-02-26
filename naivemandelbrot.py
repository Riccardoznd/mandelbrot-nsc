"""
Mandelbrot Set Generator naive implementation

Author: Riccardo Zanda
Course: Numerical Scientific Computing 2026

"""

#defining the region
xmin = -2.0
xmax = 1.0
ymin = -1.5
ymax = 1.5

#defining resolution
width=1024
height=1024


#defining step size to have evenly spaced vectors
x_step=(xmax-xmin)/(width-1)
y_step=(ymax-ymin)/(heigth-1)


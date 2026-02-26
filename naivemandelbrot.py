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
heigth=1024


#defining step size to have evenly spaced vectors
x_step=(xmax-xmin)/(width-1)
y_step=(ymax-ymin)/(heigth-1)

#creating empty vectors
x=[]
y=[]

#loops to generate evenly spaced vector
for i in range(width):
   x.append(xmin+(i*x_step))

for i in range(heigth):
   y.append(ymin+(i*y_step))

#create c=x_val*y_val*1j

C=[]

#outer loop to loop though each imaginary value
#inner loop to iterate through to real value 

for y_val in y:
    row=[]
    for x_val in x:
        c=x_val + (y_val*1j)
        row.append(c)
    C.append(row)

    

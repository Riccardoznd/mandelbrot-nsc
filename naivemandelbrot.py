"""
Mandelbrot Set Generator naive implementation

Author: Riccardo Zanda
Course: Numerical Scientific Computing 2026

"""

import matplotlib.pyplot as plt
import time

#starting the timer
start_time=time.time()


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

#i definine max number of iteration
max_iter=100

Z=[] #this will be the map

for row in C:
    row_values = []
    for c in row:
        z = 0
        n = 0
        
        # Run the actual loop
        for n in range(max_iter):
            z = z**2 + c
            
            # Check for the threshold 
            if abs(z) > 2:
                row_values.append(n)
                break
        else:
            row_values.append(max_iter)
            
    Z.append(row_values)

#ending the timer
end_time=time.time()

#now i calculate the difference between strt and end, in order to measure performance
duration=end_time-start_time
print(f"Code Time: {duration:.4f} seconds")

#i visualize the result 
plt.figure(figsize=(10, 10))
plt.imshow(Z, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap='hot')
plt.title('Mandelbrot Set')
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
plt.show()




    

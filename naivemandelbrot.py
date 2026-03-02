"""
Mandelbrot Set Generator naive implementation

Author: Riccardo Zanda
Course: Numerical Scientific Computing 2026

"""

import matplotlib.pyplot as plt
import time
import statistics

def benchmark(func, *args, n_runs=3):
    times = []
    result = None
    
    for _ in range(n_runs):
        t0 = time.perf_counter()           # Start timer
        result = func(*args)               # Call the function
        t1 = time.perf_counter()           # End timer
        times.append(t1 - t0)              # Store elapsed time
    
    median_t = statistics.median(times)
    
    print(f"Median: {median_t:.4f}s "
          f"(min={min(times):.4f}, max={max(times):.4f})")
    
    return median_t, result


#defining the region
xmin = -2.0
xmax = 1.0
ymin = -1.5
ymax = 1.5

#defining resolution
width=1024
height=1024
max_iter=100

def naive_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):

    #defining step size to have evenly spaced vectors
    x_step=(xmax-xmin)/(width-1)
    y_step=(ymax-ymin)/(height-1)

    #creating empty vectors
    x=[]
    y=[]

    #loops to generate evenly spaced vector
    for i in range(width):
        x.append(xmin+(i*x_step))

    for i in range(height):
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
    return Z



t,Z=benchmark(naive_mandelbrot,xmin, xmax, ymin, ymax, width, height, max_iter)

#i visualize the result 
plt.figure(figsize=(10, 10))
plt.imshow(Z, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap='hot')
plt.title('Mandelbrot Set')
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
plt.show()




    
	